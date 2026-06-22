# 分层块采样数据划分设计文档

## 1. 概述

### 1.1 问题背景

当前半监督学习采用时序划分（`split_three_way`），导致训练集和测试集的时间分布不一致：

- 训练集 (2015-2018): AQI 均值 ≈ 108
- 测试集 (2022-2024): AQI 均值 ≈ 77

模型学习到"时间位置→AQI"的虚假相关性，而非"过去14天数据→未来3天数据"的真实规律。

### 1.2 设计目标

1. 消除时间分布偏差
2. 保持序列完整性（14天输入→3天输出）
3. 确保重叠样本在同一集合
4. 模型学习真实规律而非虚假相关

## 2. 核心设计

### 2.1 数据流

```
原始数据 (3652天)
        ↓
滑动窗口创建样本 (3636个样本)
        ↓
按块分组 (约121个块，每块30天)
        ↓
按块的中心日期分层 (约40层)
        ↓
每层内随机采样块
        ↓
分配到三个集合:
  - Labeled: 40% 的块
  - Unlabeled: 40% 的块
  - Test: 20% 的块
        ↓
每个集合包含完整的样本（保持序列完整性）
```

### 2.2 滑动窗口

| 参数 | 值 | 说明 |
|------|-----|------|
| `seq_length` | 14 | 输入序列长度（天） |
| `prediction_days` | 3 | 预测天数 |
| `stride` | 1 | 滑动步长（天） |

**样本数计算**：
- 原始数据：3652 天
- 每个样本需要：14 + 3 = 17 天
- 滑动步长：1 天
- 样本数：3652 - 17 + 1 = 3636 个样本

### 2.3 块划分

**块大小**：30 天（一个月）

**块定义**：
```python
def create_blocks(n_samples, block_size=30):
    """
    将连续样本分组为块
    块 i 包含样本 [i*block_size, (i+1)*block_size)
    """
    blocks = []
    for start in range(0, n_samples, block_size):
        end = min(start + block_size, n_samples)
        blocks.append((start, end))
    return blocks
```

**块数**：3636 / 30 ≈ 121 个块

### 2.4 分层策略

**分层维度**：年份 + 季节组合

```python
def get_stratum_label(date):
    """获取分层标签：年份+季节"""
    year = date.year
    season = (date.month % 12 + 3) // 3  # 1-4
    return f"{year}_Q{season}"
```

**层数**：约 40 层（10年 × 4季节）

**分层逻辑**：
- 每个块按其中心日期分层
- 中心日期 = 块内样本的起始日期的中位数

### 2.5 采样算法

```python
def stratified_block_sampling(X, y, dates, block_size=30, ratios=(0.4, 0.4, 0.2)):
    """
    分层块采样算法

    Args:
        X: (N, 14, 9) 输入序列
        y: (N, 3, 7) 输出标签
        dates: (N,) 日期序列
        block_size: 块大小（天）
        ratios: (labeled, unlabeled, test) 比例

    Returns:
        (X_labeled, y_labeled), (X_unlabeled, y_unlabeled), (X_test, y_test)
    """
    # 1. 创建块
    blocks = create_blocks(len(X), block_size)

    # 2. 按块的中心日期分层
    block_strata = []
    for start, end in blocks:
        center_idx = (start + end) // 2
        center_date = dates[center_idx]
        stratum = get_stratum_label(center_date)
        block_strata.append(stratum)

    # 3. 按层分组
    stratum_groups = defaultdict(list)
    for i, stratum in enumerate(block_strata):
        stratum_groups[stratum].append(i)

    # 4. 每层内随机采样块
    labeled_blocks = []
    unlabeled_blocks = []
    test_blocks = []

    for stratum, block_indices in stratum_groups.items():
        np.random.shuffle(block_indices)
        n = len(block_indices)
        n_label = int(n * ratios[0])
        n_unlabel = int(n * ratios[1])

        labeled_blocks.extend(block_indices[:n_label])
        unlabeled_blocks.extend(block_indices[n_label:n_label + n_unlabel])
        test_blocks.extend(block_indices[n_label + n_unlabel:])

    # 5. 收集样本
    def collect_samples(block_list):
        indices = []
        for block_idx in block_list:
            start, end = blocks[block_idx]
            indices.extend(range(start, end))
        return X[indices], y[indices]

    return (
        collect_samples(labeled_blocks),
        collect_samples(unlabeled_blocks),
        collect_samples(test_blocks),
    )
```

## 3. 处理边界情况

### 3.1 样本数过少的层

**策略**：合并到相邻层

```python
def merge_small_strata(stratum_groups, min_samples=10):
    """
    如果某层样本数 < min_samples，合并到相邻季节
    """
    merged = defaultdict(list)
    for stratum, indices in stratum_groups.items():
        if len(indices) < min_samples:
            # 合并到相邻季节
            year, season = stratum.split('_Q')
            season = int(season)
            # 选择相邻季节
            adjacent_season = season + 1 if season < 4 else season - 1
            adjacent_stratum = f"{year}_Q{adjacent_season}"
            merged[adjacent_stratum].extend(indices)
        else:
            merged[stratum].extend(indices)
    return merged
```

### 3.2 块边界效应

**问题**：块边界的样本可能有特殊模式

**缓解**：使用重叠块或随机偏移

```python
def create_overlapping_blocks(n_samples, block_size=30, overlap=5):
    """
    创建重叠块，减少边界效应
    """
    blocks = []
    start = 0
    while start < n_samples:
        end = min(start + block_size, n_samples)
        blocks.append((start, end))
        start += block_size - overlap
    return blocks
```

## 4. 验证指标

### 4.1 时间分布一致性

```python
def verify_time_distribution(X_labeled, X_unlabeled, X_test, dates):
    """
    验证三个集合的时间分布是否一致
    """
    # 计算每个集合的年份分布
    labeled_years = [dates[i].year for i in range(len(X_labeled))]
    unlabeled_years = [dates[i].year for i in range(len(X_unlabeled))]
    test_years = [dates[i].year for i in range(len(X_test))]

    # 统计各年份样本数
    # 应该呈现相似的分布
```

### 4.2 AQI分布一致性

```python
def verify_aqi_distribution(X_labeled, X_unlabeled, X_test):
    """
    验证三个集合的AQI分布是否一致
    """
    # 计算每个集合的AQI均值和方差
    # 应该接近
```

### 4.3 样本独立性

```python
def verify_sample_independence(X_labeled, X_unlabeled, X_test, blocks):
    """
    验证同一块的样本在同一集合
    """
    # 检查是否有重叠样本跨越不同集合
```

## 5. 实现计划

### 5.1 修改文件

1. `src/air_quality/data/processor.py`
   - 添加 `create_blocks` 函数
   - 添加 `get_stratum_label` 函数
   - 添加 `stratified_block_sampling` 函数
   - 修改 `split_three_way` 方法或添加新方法

2. `scripts/train.py`
   - 调用新的数据划分方法
   - 传递日期信息用于分层

3. `src/air_quality/config/settings.py`
   - 添加块大小、重叠等配置参数

### 5.2 测试计划

1. **单元测试**：验证分层采样逻辑
2. **集成测试**：验证端到端流程
3. **效果验证**：对比修改前后的预测结果

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 块边界效应 | 块边界的样本可能有特殊模式 | 使用重叠块或随机偏移 |
| 分层不均 | 某些层样本过少 | 合并到相邻层 |
| 计算开销 | 滑动窗口增加样本数 | 使用批量处理 |
| 时序性损失 | 打乱样本顺序可能损失部分时序信息 | 使用块划分保留局部时序性 |

## 7. 预期效果

### 7.1 消除时间偏差

- 训练集和测试集都包含所有时间段的数据
- 模型学习"过去14天数据→未来3天数据"的映射
- 而不是"时间位置→AQI"的映射

### 7.2 提升泛化能力

- 模型在所有时间段上都有训练样本
- 测试时不会遇到"未见过的时间段"
- 预测结果更稳定

### 7.3 保持时序性

- 每个样本的序列完整性保持
- 块划分保留局部时序模式
- 模型仍能学习短期依赖关系

## 8. 总结

本设计通过分层块采样解决了半监督学习中的时间趋势问题：

1. **分层采样**：确保时间分布一致
2. **块划分**：保持重叠样本在同一集合
3. **滑动窗口**：保持序列完整性

最终目标是让模型学习真实规律（过去数据→未来数据），而非虚假相关（时间位置→AQI）。
