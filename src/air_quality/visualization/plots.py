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


def plot_training_history(history: dict, save_path: str = None, show: bool = False):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if 'train_loss' in history and 'test_loss' in history:
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

    # 最佳epoch标注
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
