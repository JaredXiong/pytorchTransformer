"""
绘图模块 - 训练可视化和预测对比
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg')  # 非交互式后端
import os
from typing import List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def _draw_empty_panel(ax, message: str) -> None:
    """在子图上渲染一条友好提示并关闭坐标轴（避免空白图）。"""
    ax.text(0.5, 0.5, message, ha='center', va='center',
            transform=ax.transAxes, fontsize=11, color='gray', wrap=True)
    ax.set_title(message.split('\n')[0][:40])
    ax.axis('off')


def _plot_semi_history(fig, axes, history: dict) -> None:
    """渲染半监督 / 预训练-微调 训练历史。

    布局（1×3，与全监督 1×2 信息对等）：
      - 左：各阶段 Train/Test Loss（按阶段拼接 x 轴 + 阶段背景色 + 分隔线）
      - 中：学习率曲线（按阶段拼接，与全监督右图等价）
      - 右：反平滑检测信号（每 epoch 被标记的污染物数）+ 伪标签保留率柱状图

    三阶段的 train_loss 来自不同 model 实例，因此 train/test 曲线**不**跨阶段
    连续拼接，而是按阶段分别绘制并标注阶段背景色；x 轴 epoch 仍连续以便阅读。
    """
    pretrain_losses = history.get('pretrain_losses', [])
    teacher_losses = history.get('teacher_losses', [])
    student_losses = history.get('student_losses', [])
    # 反平滑聚合后的分阶段曲线（由 SemiSupervisedTrainer._record_phase_history 写入）
    teacher_test = history.get('teacher_test_loss', [])
    student_test = history.get('student_test_loss', [])
    teacher_lr = history.get('teacher_lr', [])
    student_lr = history.get('student_lr', [])

    pretrain_n = len(pretrain_losses)
    teacher_n = len(teacher_losses)
    student_n = len(student_losses)
    total_n = pretrain_n + teacher_n + student_n

    ax_loss, ax_lr, ax_aux = axes[0], axes[1], axes[2]

    # ----- 左图：Loss（按阶段） -----
    if total_n == 0:
        _draw_empty_panel(ax_loss, 'No loss data\n(各阶段均未产生训练损失)')
    else:
        # 阶段区间: [(name, color, start_epoch_1based, n)]
        segments = [
            ('Pretrain', '#1f77b4', pretrain_losses, []),
            ('Teacher', '#ff7f0e', teacher_losses, teacher_test),
            ('Student', '#2ca02c', student_losses, student_test),
        ]
        cursor = 0  # 已分配的 epoch 数（1-based 起点 = cursor）
        for name, color, train_vals, test_vals in segments:
            n = len(train_vals)
            if n == 0:
                continue
            start = cursor  # 0-based offset
            x = range(start + 1, start + n + 1)
            ax_loss.plot(x, train_vals, label=f'{name} Train', marker='o',
                         markersize=3, color=color)
            # 同阶段 test loss（若存在且长度一致）用虚线叠加
            if len(test_vals) == n:
                ax_loss.plot(x, test_vals, label=f'{name} Val', marker='x',
                             markersize=3, color=color, linestyle='--', alpha=0.6)
            ax_loss.axvspan(start + 0.5, start + n + 0.5, alpha=0.08, color=color)
            cursor += n
        # 阶段分隔线（仅在有 ≥2 个非空阶段时画）
        nonempty_bounds = []
        c = 0
        for vals in (pretrain_losses, teacher_losses, student_losses):
            if len(vals) > 0:
                c += len(vals)
                nonempty_bounds.append(c)
        for b in nonempty_bounds[:-1]:
            ax_loss.axvline(x=b + 0.5, color='gray', linestyle=':',
                            linewidth=0.8, alpha=0.6)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        title = 'Training History - Loss (Semi-Supervised)'
        best_loss = history.get('best_loss')
        if best_loss is not None and best_loss != float('inf'):
            title += f'  (Best: {best_loss:.4f})'
        ax_loss.set_title(title)
        ax_loss.legend(loc='upper right', fontsize=8)
        ax_loss.grid(True, alpha=0.3)

    # ----- 中图：学习率曲线（按阶段拼接） -----
    lr_segments = [
        ('Pretrain', '#1f77b4', []),  # pretrain 用固定 lr，这里不绘制逐 epoch
        ('Teacher', '#ff7f0e', teacher_lr),
        ('Student', '#2ca02c', student_lr),
    ]
    has_lr = any(len(v) > 0 for _, _, v in lr_segments)
    if not has_lr:
        _draw_empty_panel(ax_lr, 'No learning rate data\n(OneCycleLR 未记录)')
    else:
        cursor = 0
        for name, color, lr_vals in lr_segments:
            n = len(lr_vals)
            if n == 0:
                continue
            x = range(cursor + 1, cursor + n + 1)
            ax_lr.plot(x, lr_vals, label=f'{name} LR', color=color, linewidth=1.5)
            cursor += n
        ax_lr.set_xlabel('Epoch')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('Learning Rate Schedule (by Phase)')
        ax_lr.grid(True, alpha=0.3)
        ax_lr.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        ax_lr.legend(loc='upper right', fontsize=8)

    # ----- 右图：反平滑信号 + 伪标签保留率 -----
    # 反平滑标记：每 epoch 被标记为过度平滑的污染物数（跨阶段顺序拼接）
    flags = history.get('smoothing_flags_per_epoch', [])
    pseudo_rates = history.get('pseudo_label_rate', [])
    has_flags = len(flags) > 0
    has_pseudo = len(pseudo_rates) > 0

    if not has_flags and not has_pseudo:
        _draw_empty_panel(ax_aux, 'No smoothing / pseudo-label data')
        return

    # 双 y 轴：左=被标记污染物数（折线），右=伪标签保留率（柱状）
    if has_flags:
        n_flagged = [f.get('n_flagged', 0) for f in flags]
        # 总特征数（取任一非空 flags 记录的长度）
        total_feat = 7
        for f in flags:
            fl = f.get('flags', {})
            if fl:
                total_feat = len(fl)
                break
        ax_aux.plot(range(1, len(n_flagged) + 1), n_flagged,
                    color='crimson', marker='.', markersize=4,
                    label=f'Over-smooth flagged (/{total_feat})')
        ax_aux.set_xlabel('Epoch')
        ax_aux.set_ylabel('# Flagged pollutants', color='crimson')
        ax_aux.tick_params(axis='y', labelcolor='crimson')
        ax_aux.set_ylim(-0.3, total_feat + 0.3)
        ax_aux.set_title('Anti-Smoothing Signal & Pseudo-Label Rate')
        ax_aux.grid(True, alpha=0.2)

        if has_pseudo:
            ax_aux2 = ax_aux.twinx()
            # 伪标签阶段数通常 << epoch 数，用右侧短柱表示
            ax_aux2.bar(range(1, len(pseudo_rates) + 1), pseudo_rates,
                        color='steelblue', alpha=0.4, edgecolor='navy',
                        width=0.6, label='Pseudo-label rate')
            ax_aux2.set_ylabel('Pseudo-label retention rate', color='navy')
            ax_aux2.tick_params(axis='y', labelcolor='navy')
            ax_aux2.set_ylim(0, 1)
            # 合并图例
            h1, l1 = ax_aux.get_legend_handles_labels()
            h2, l2 = ax_aux2.get_legend_handles_labels()
            ax_aux.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7)
        else:
            ax_aux.legend(loc='upper right', fontsize=8)
    else:
        # 仅有伪标签率
        ax_aux.bar(range(1, len(pseudo_rates) + 1), pseudo_rates,
                   color='steelblue', alpha=0.7, edgecolor='black')
        ax_aux.set_xlabel('Phase')
        ax_aux.set_ylabel('Pseudo-label retention rate')
        ax_aux.set_title('Pseudo-Label Rate (by Phase)')
        ax_aux.set_ylim(0, 1)
        ax_aux.grid(True, alpha=0.3, axis='y')


def plot_training_history(history: dict, save_path: str = None, show: bool = False):
    """绘制训练历史。

    自动识别两种 history 格式并渲染**信息对等**的图表：
    - 全监督 (``train_loss`` + ``test_loss`` + ``lr``): 1×2 — 训练/验证 loss + 学习率
    - 半监督 (``teacher_losses`` + ``student_losses`` + 可选 ``pretrain_losses`` +
      ``pseudo_label_rate`` + ``<phase>_lr``/``<phase>_test_loss``/
      ``smoothing_flags_per_epoch``): 1×3 — 分阶段 loss + 学习率 + 反平滑信号/伪标签率

    半监督字段缺失时会友好降级（显示提示文字而非空白图），修复"字段不匹配导致空白图表"。
    """
    has_pretrain = bool(history.get('pretrain_losses'))
    has_semi = bool(history.get('teacher_losses') or history.get('student_losses'))

    if has_semi or has_pretrain:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        _plot_semi_history(fig, axes, history)
    elif 'train_loss' in history and 'test_loss' in history:
        # === 全监督历史 ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        train_losses = history.get('train_loss', [])
        test_losses = history.get('test_loss', [])
        epochs = range(1, len(train_losses) + 1)

        axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
        axes[0].plot(epochs, test_losses, label='Test Loss', marker='s', markersize=3)

        if 'best_epoch' in history:
            best_epoch = history['best_epoch']
            axes[0].axvline(x=best_epoch + 1, color='r', linestyle='--',
                            label=f'Best Epoch ({best_epoch + 1})')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if 'best_loss' in history and history['best_loss'] != float('inf'):
            axes[0].set_title(f'Training History - Loss (Best: {history["best_loss"]:.4f})')

        # 学习率曲线
        if 'lr' in history and len(history['lr']) > 0:
            lr_values = history['lr']
            lr_epochs = range(1, len(lr_values) + 1)
            axes[1].plot(lr_epochs, lr_values, color='green', linewidth=1.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].grid(True, alpha=0.3)
            axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        else:
            _draw_empty_panel(axes[1], 'No learning rate data')
    else:
        # === 未知格式：友好提示 ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        _draw_empty_panel(axes[0],
                          'Unsupported history format\nkeys: ' + ', '.join(history.keys()))
        axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_prediction_comparison(y_true: np.ndarray, y_pred: np.ndarray,dates: List[str] = None, feature_names: List[str] = None,save_path: str = None, show: bool = False):
    """绘制预测值与真实值对比图"""
    feature_names = feature_names or ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Month', 'Season']
    n_features = min(y_true.shape[-1], len(feature_names))

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        true_vals = y_true[..., i].flatten()
        pred_vals = y_pred[..., i].flatten()
        x_vals = range(len(true_vals))

        ax.plot(x_vals, true_vals, label='True', alpha=0.8, linewidth=1.5)
        ax.plot(x_vals, pred_vals, label='Predicted', alpha=0.8, linewidth=1.5, linestyle='--')

        ax.set_title(f'{feature_names[i]}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_feature_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                            feature_idx: int = 0, feature_name: str = 'AQI',
                            dates: List[str] = None, save_path: str = None, show: bool = False):
    """绘制单个特征的详细对比图（含置信区间）"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    true_vals = y_true[..., feature_idx]
    pred_vals = y_pred[..., feature_idx]

    # 上图：时序对比
    x_vals = range(len(true_vals))
    axes[0].plot(x_vals, true_vals, label='True', color='blue', alpha=0.8, linewidth=1.5)
    axes[0].plot(x_vals, pred_vals, label='Predicted', color='red', alpha=0.8, linewidth=1.5, linestyle='--')

    # 添加置信区间（基于预测误差）
    error = np.abs(true_vals - pred_vals)
    mean_error = np.mean(error)
    axes[0].fill_between(x_vals, pred_vals - mean_error, pred_vals + mean_error,
                        alpha=0.2, color='red', label=f'Error Range ({mean_error:.2f})')

    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel(feature_name)
    axes[0].set_title(f'{feature_name} - True vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：误差分布
    error = true_vals - pred_vals
    axes[1].hist(error.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', label='Zero Error')
    axes[1].axvline(x=np.mean(error), color='green', linestyle='--',
                    label=f'Mean Error ({np.mean(error):.2f})')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{feature_name} - Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_metrics(metrics: dict, save_path: str = None, show: bool = False):
    """绘制评估指标柱状图"""
    main_metrics = ['mse', 'rmse', 'mae', 'r2']
    feature_metrics = {}

    for key, value in metrics.items():
        for m in main_metrics:
            if key.endswith(f'_{m}'):
                feature = key.replace(f'_{m}', '')
                if feature not in feature_metrics:
                    feature_metrics[feature] = {}
                feature_metrics[feature][m] = value

    n_features = len(feature_metrics)
    if n_features == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        features = list(feature_metrics.keys())
        values = [feature_metrics[f].get(metric, 0) for f in features]

        bars = ax.bar(features, values, color='steelblue', alpha=0.7)
        ax.set_title(f'{metric.upper()} by Feature')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_backtest_results(y_true: np.ndarray, y_pred: np.ndarray,
                          dates: list = None, feature_names: list = None,
                          save_path: str = None, show: bool = False,
                          model_type: str = None,
                          y_true_raw: np.ndarray = None) -> str:
    """绘制回测结果：每个污染物生成一张独立图表，保存到模型专属文件夹。

    Args:
        y_true: shape (N, prediction_days, 7) —— 经过预处理裁剪/缩放的真值
        y_pred: shape (N, prediction_days, 7)
        dates: list of length N，每个元素是 prediction_days 个日期字符串
        feature_names: 列名（默认 7 个污染物）
        save_path: 保存目录（若为 None 则使用 outputs/figures/<model_type>/）
        show: 是否 plt.show()
        model_type: 模型名称，用于子文件夹命名
        y_true_raw: shape (N, prediction_days, 7) —— 原始未裁剪的真值（可选）。
                    若提供，则作为"真实值"曲线绘制，确保 y 轴反映数据的真实范围。
                    若为 None，则回退到 y_true（可能已被 _handle_outliers 裁剪）。

    Returns:
        保存目录路径
    """
    feature_names = feature_names or ['AQI', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    n_features = min(y_true.shape[-1], len(feature_names))

    # 优先使用原始未裁剪的真值，让 y 轴反映真实数据范围（不被 _handle_outliers 强行截断）
    if y_true_raw is not None:
        true_flat = y_true_raw[..., :n_features].reshape(-1, n_features)
    else:
        true_flat = y_true[..., :n_features].reshape(-1, n_features)
    pred_flat = y_pred[..., :n_features].reshape(-1, n_features)

    # 解析日期
    use_dates = False
    x_dates = None
    if dates is not None:
        import pandas as pd
        flat_dates = []
        for window_dates in dates:
            flat_dates.extend(window_dates)
        x_dates = pd.to_datetime(flat_dates)
        use_dates = True

    # 确定保存目录
    if save_path is not None:
        out_dir = save_path if os.path.isdir(save_path) else os.path.dirname(save_path)
    else:
        out_dir = os.path.join('outputs', 'figures', model_type or 'default')
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_features):
        fig, ax = plt.subplots(figsize=(16, 6), facecolor='white')
        ax.set_facecolor('white')

        if use_dates:
            ax.plot(x_dates, true_flat[:, i], label='真实值', alpha=0.8, linewidth=1.2)
            ax.plot(x_dates, pred_flat[:, i], label='预测值', alpha=0.8, linewidth=1.2, linestyle='--')
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            fig.autofmt_xdate(rotation=30, ha='right')
            ax.set_xlabel('日期')
        else:
            x_vals = np.arange(len(true_flat))
            ax.plot(x_vals, true_flat[:, i], label='真实值', alpha=0.8, linewidth=1.2)
            ax.plot(x_vals, pred_flat[:, i], label='预测值', alpha=0.8, linewidth=1.2, linestyle='--')
            ax.set_xlabel('时间步')

        ax.set_ylabel(feature_names[i])
        ax.set_title(feature_names[i], fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.2, color='gray')

        # 去掉灰色边框
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

        plt.tight_layout()

        fig_path = os.path.join(out_dir, f'{feature_names[i].replace(".", "_")}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        if show:
            fig.show()

    return out_dir
