"""
训练脚本 - 端到端训练流程

用法：
    python -m scripts.train                        # 训练默认 Hybrid 模型
    python -m scripts.train --model lstm           # 训练指定模型
    python -m scripts.train --epochs 50            # 自定义训练轮数
    python -m scripts.train --model transformer --loss-type mse_antismooth  # 反平滑训练
"""
import argparse
import logging
import os
import sys
import time
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch

# 让 `python -m scripts.train` 能 import air_quality
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor, split_data, get_device
from air_quality.models import create_model
from air_quality.training import ModelTrainer
from air_quality.inference import predict_air_quality, format_prediction_result
from air_quality.visualization import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """创建必要的目录"""
    os.makedirs(os.path.dirname(config.file.model_save_path), exist_ok=True)
    os.makedirs(config.file.figures_dir, exist_ok=True)


def prepare_data():
    """加载数据、构建序列、划分训练/测试集。

    Returns:
        (train_loader, test_loader, scaler, features, dates, data, raw_data, processor)
        - data: scaled (N, 9) — 前 7 缩放污染物 + 后 2 原始 month/season
        - raw_data: 原始尺度 (N, 9) — 用于 forecast 演示
    """
    logger.info(f"正在加载数据: {config.data.data_file}")
    processor = AirQualityDataProcessor(
        seq_length=config.data.seq_length,
        prediction_days=config.data.prediction_days,
        rolling_windows=config.data.rolling_windows,
        output_size=config.model.output_size,
        scaler_on_uncapped=config.data.scaler_on_uncapped,
    )
    data, scaler, features, dates = processor.load_and_preprocess_data(config.data.data_file)
    logger.info(f"数据加载完成，特征数量: {len(features)}, 数据点数量: {len(data)}")

    # 构造 raw data：原始污染物 + month/season
    raw_df = pd.read_excel(config.data.data_file, engine='openpyxl')
    date_col = 'pubtime' if 'pubtime' in raw_df.columns else 'date'
    raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
    raw_df['month'] = raw_df[date_col].dt.month
    raw_df['season'] = (raw_df[date_col].dt.month % 12 + 3) // 3
    column_mapping = processor._create_column_mapping(raw_df.columns)
    raw_df = raw_df.rename(columns=column_mapping)
    pollutant_cols = [c for c in processor.feature_columns if c in raw_df.columns and c not in ('month', 'season')]
    raw_data = raw_df[pollutant_cols + ['month', 'season']].fillna(
        raw_df[pollutant_cols + ['month', 'season']].mean()
    ).values

    X, y = processor.create_sequences(data)
    X_train, X_test, y_train, y_test = split_data(X, y, config.data.train_split_ratio)
    logger.info(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, scaler, features, dates, data, raw_data, processor


def save_artifacts(model, scaler) -> None:
    """保存模型权重与标准化器"""
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_type': model.__class__.__name__,
        },
        config.file.model_save_path,
    )
    joblib.dump(scaler, config.file.scaler_path)
    logger.info(f"模型已保存到: {config.file.model_save_path}")
    logger.info(f"标准化器已保存到: {config.file.scaler_path}")


def log_training_summary(history: dict) -> None:
    """打印训练摘要"""
    logger.info("\n训练总结:")
    logger.info(f"  总训练轮次: {history.get('epochs_trained', 0)}")
    logger.info(f"  最佳轮次: {history.get('best_epoch', 0) + 1}")
    logger.info(f"  最佳验证损失: {history.get('best_loss', float('inf')):.6f}")
    if history.get('smoothing_stopped'):
        logger.info("  ⚠ 训练因反平滑检测触发提前停止")
    flags = history.get('smoothing_flags_per_epoch', [])
    if flags:
        last = flags[-1]
        logger.info(f"  最终 epoch 过度平滑标记: {last['n_flagged']}/{len(last.get('flags', {}))}")


def run_prediction_example(raw_data, dates) -> None:
    """运行预测示例（生产模式：forecast）"""
    if len(raw_data) < config.data.seq_length:
        logger.warning(
            f"数据长度({len(raw_data)})不足以创建预测序列({config.data.seq_length})"
        )
        return

    logger.info("执行预测示例")
    sample_input = raw_data[-config.data.seq_length:]

    last_date = pd.to_datetime(dates.iloc[-1])
    future_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d')
        for i in range(config.data.prediction_days)
    ]

    prediction_result = predict_air_quality(
        input_sequence=sample_input,
        num_days=config.data.prediction_days,
        future_dates=future_dates,
    )
    print(format_prediction_result(prediction_result, show_details=True))


def train_single_model(model_type: str, train_loader, test_loader, device) -> Dict[str, Any]:
    """训练单个模型并返回结果"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"训练模型: {model_type}")
    logger.info(f"{'=' * 60}")

    # 模型工厂参数
    model_kwargs = {'dropout': config.model.dropout}
    if model_type == 'transformer':
        # 把 transformer 专用反平滑参数传入
        model_kwargs.update({
            'residual_weight': config.model.transformer_residual_weight,
            'feature_scale_init': config.model.transformer_feature_scale_init,
            'ffn_mult': config.model.transformer_ffn_mult,
            'norm_first': config.model.transformer_norm_first,
        })

    model = create_model(
        model_type,
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        **model_kwargs,
    ).to(device)

    # 反平滑损失 kwargs
    loss_kwargs = {
        'delta': config.training.delta,
        'lambda_var': config.training.lambda_var,
        'lambda_diff': config.training.lambda_diff,
        'tau_var': config.training.tau_var,
        'tau_diff': config.training.tau_diff,
        'lambda_warmup_epochs': config.training.lambda_warmup_epochs,
    }

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        best_model_path=config.file.model_save_path,
        loss_type=config.training.loss_type,
        early_stop_patience=config.training.early_stop_patience,
        gradient_clip=config.training.gradient_clip,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        loss_kwargs=loss_kwargs,
        detect_smoothing=config.training.detect_smoothing,
        smoothing_threshold=config.training.smoothing_threshold,
        smoothing_stop_patience=config.training.smoothing_stop_patience,
    )

    start = time.time()
    trainer.train(num_epochs=config.training.epochs)
    elapsed = time.time() - start

    return {
        'model': model,
        'history': trainer.history,
        'training_time': elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Air Quality Time-Series Training')
    parser.add_argument(
        '--model', default='hybrid',
        choices=['transformer', 'lstm', 'cnn', 'hybrid'],
        help='要训练的模型类型 (默认: hybrid)',
    )
    parser.add_argument('--batch-size', type=int, help='覆盖batch size')
    parser.add_argument('--lr', type=float, help='覆盖学习率')
    parser.add_argument('--epochs', type=int, help='覆盖训练轮数')
    # 损失与正则化
    parser.add_argument(
        '--loss-type', default=None,
        choices=['mse', 'huber', 'mae', 'smooth_l1', 'mse_antismooth'],
        help='损失函数类型（默认从 config 读取）',
    )
    parser.add_argument('--delta', type=float, help='Huber 损失 delta')
    parser.add_argument('--dropout', type=float, help='模型 dropout')
    parser.add_argument('--weight-decay', type=float, help='L2 正则化系数')
    parser.add_argument('--early-stop', type=int, dest='early_stop', help='早停耐心值')
    parser.add_argument('--gradient-clip', type=float, help='梯度裁剪阈值')
    # 反平滑检测
    parser.add_argument(
        '--detect-smoothing', dest='detect_smoothing', action='store_true', default=None,
        help='启用反平滑检测作为额外早停信号',
    )
    parser.add_argument(
        '--no-detect-smoothing', dest='detect_smoothing', action='store_false', default=None,
        help='关闭反平滑检测',
    )
    parser.add_argument('--smoothing-threshold', type=float, help='反平滑阈值 (std/diff_ratio < thresh 视为平滑)')
    parser.add_argument('--smoothing-stop-patience', type=int, help='反平滑早停耐心值')
    # 反平滑损失权重
    parser.add_argument('--lambda-var', dest='lambda_var', type=float, help='方差下界惩罚权重')
    parser.add_argument('--lambda-diff', dest='lambda_diff', type=float, help='一阶差分下界惩罚权重')
    parser.add_argument('--tau-var', dest='tau_var', type=float, help='方差下界阈值（pred/true std 最小比）')
    parser.add_argument('--tau-diff', dest='tau_diff', type=float, help='一阶差分下界阈值')
    # 数据与推理
    parser.add_argument(
        '--no-cap-training-data', dest='no_cap_training_data', action='store_true', default=False,
        help='使用旧行为：scaler 在裁剪后数据上 fit（不推荐）',
    )
    parser.add_argument(
        '--no-soft-clip', dest='no_soft_clip', action='store_true', default=False,
        help='关闭推理软饱和（恢复硬裁剪）',
    )
    # 半监督与 VMD
    parser.add_argument(
        '--semi-supervised', dest='semi_supervised', action='store_true',
        help='启用半监督训练（仅 vmd_cnn_bilstm_attention 模型支持）',
    )
    parser.add_argument('--vmd-k', type=int, help='VMD 模态数（默认从 config 读取）')
    parser.add_argument('--no-vmd', dest='no_vmd', action='store_true', help='关闭 VMD 分解')
    parser.add_argument(
        '--pseudo-threshold', dest='pseudo_threshold', type=float,
        help='伪标签置信度阈值',
    )
    parser.add_argument(
        '--teacher-epochs', dest='teacher_epochs', type=int,
        help='Teacher 训练轮数',
    )
    parser.add_argument(
        '--student-epochs', dest='student_epochs', type=int,
        help='Student 训练轮数',
    )
    # 预训练-微调(VMD 半监督基础上)
    parser.add_argument(
        '--pretrain', dest='pretrain', action='store_true',
        help='启用无监督预训练-有监督微调范式(需同时 --semi-supervised)',
    )
    parser.add_argument(
        '--pretrain-epochs', dest='pretrain_epochs', type=int, default=None,
        help='预训练轮数(默认从 config 读取)',
    )
    parser.add_argument(
        '--pretrain-mask-ratio', dest='pretrain_mask_ratio', type=float, default=None,
        help='掩码比例 0~1',
    )
    parser.add_argument(
        '--pretrain-lr', dest='pretrain_lr', type=float, default=None,
        help='预训练学习率',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        logger.info("空气质量预测系统启动")
        setup_directories()

        # 把 CLI 覆盖写入 config
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.lr:
            config.training.learning_rate = args.lr
        if args.epochs:
            config.training.epochs = args.epochs
        if args.loss_type is not None:
            config.training.loss_type = args.loss_type
        if args.delta is not None:
            config.training.delta = args.delta
        if args.dropout is not None:
            config.model.dropout = args.dropout
        if args.weight_decay is not None:
            config.training.weight_decay = args.weight_decay
        if args.early_stop is not None:
            config.training.early_stop_patience = args.early_stop
        if args.gradient_clip is not None:
            config.training.gradient_clip = args.gradient_clip
        if args.detect_smoothing is not None:
            config.training.detect_smoothing = args.detect_smoothing
        if args.smoothing_threshold is not None:
            config.training.smoothing_threshold = args.smoothing_threshold
        if args.smoothing_stop_patience is not None:
            config.training.smoothing_stop_patience = args.smoothing_stop_patience
        if args.lambda_var is not None:
            config.training.lambda_var = args.lambda_var
        if args.lambda_diff is not None:
            config.training.lambda_diff = args.lambda_diff
        if args.tau_var is not None:
            config.training.tau_var = args.tau_var
        if args.tau_diff is not None:
            config.training.tau_diff = args.tau_diff
        if args.no_cap_training_data:
            config.data.scaler_on_uncapped = False
        if args.no_soft_clip:
            config.prediction.soft_clip = False

        # 应用 VMD / 半监督 CLI 覆盖
        if args.no_vmd:
            config.vmd.enabled = False
        if args.vmd_k is not None:
            config.vmd.K = args.vmd_k
        if args.semi_supervised:
            config.semi.enabled = True
        if args.pseudo_threshold is not None:
            config.semi.pseudo_confidence_threshold = args.pseudo_threshold
        if args.teacher_epochs is not None:
            config.semi.teacher_epochs = args.teacher_epochs
        if args.student_epochs is not None:
            config.semi.student_epochs = args.student_epochs

        # 应用预训练配置覆盖
        if args.pretrain:
            config.pretrain.enabled = True
        if args.pretrain_epochs is not None:
            config.pretrain.epochs = args.pretrain_epochs
        if args.pretrain_mask_ratio is not None:
            config.pretrain.mask_ratio = args.pretrain_mask_ratio
        if args.pretrain_lr is not None:
            config.pretrain.learning_rate = args.pretrain_lr

        train_loader, test_loader, scaler, features, dates, data, raw_data, processor = prepare_data()

        device = get_device()
        logger.info(f"使用设备: {device}")

        if config.semi.enabled:
            logger.info("=" * 60)
            logger.info("半监督模式已启用：跳过默认的全监督训练，直接进入 VMD-CNN-BiLSTM-Attention 半监督流程")
            logger.info("=" * 60)
        else:
            result = train_single_model(args.model, train_loader, test_loader, device)
            model = result['model']
            history = result['history']

        if config.semi.enabled:
            logger.info("=" * 60)
            logger.info("运行半监督训练流程 (VMD-CNN-BiLSTM-Attention)")
            logger.info("=" * 60)
            from air_quality.data.vmd import VMDDecomposer
            from air_quality.data.vmd_features import apply_vmd_to_aqi
            from air_quality.training import SemiSupervisedTrainer

            # 从 processor 拿完整序列（覆盖训练 + 测试区间）
            full_X, full_y = processor.create_sequences(data)
            if config.vmd.enabled:
                decomposer = VMDDecomposer(
                    K=config.vmd.K, alpha=config.vmd.alpha,
                    tau=config.vmd.tau, DC=config.vmd.DC,
                    init=config.vmd.init, tol=config.vmd.tol,
                )
                full_X_vmd = apply_vmd_to_aqi(full_X, decomposer)
            else:
                # 不启用 VMD 时直接使用原始 9 维特征
                full_X_vmd = full_X

            # 三段划分
            (X_lab, y_lab), (X_unl, y_unl), (X_te, y_te) = processor.split_three_way(
                full_X_vmd, full_y,
                ratios=(
                    config.semi.labeled_ratio,
                    config.semi.unlabeled_ratio,
                    config.semi.test_ratio,
                ),
            )
            logger.info(f"半监督数据集: labeled={len(X_lab)}, unlabeled={len(X_unl)}, test={len(X_te)}")

            if config.pretrain.enabled:
                from air_quality.training import PretrainFinetuneTrainer
                semi_trainer = PretrainFinetuneTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=full_X_vmd.shape[2],
                    device=str(device),
                    teacher_epochs=config.semi.teacher_epochs,
                    student_epochs=config.semi.student_epochs,
                    pseudo_confidence_threshold=config.semi.pseudo_confidence_threshold,
                    pretrain_config=config.pretrain,
                )
                logger.info("启用预训练-微调范式(Phase 0: 掩码自监督预训练)")
            else:
                semi_trainer = SemiSupervisedTrainer(
                    model_type='vmd_cnn_bilstm_attention',
                    input_size=full_X_vmd.shape[2],
                    device=str(device),
                    teacher_epochs=config.semi.teacher_epochs,
                    student_epochs=config.semi.student_epochs,
                    pseudo_confidence_threshold=config.semi.pseudo_confidence_threshold,
                )
            model, semi_metrics = semi_trainer.fit(
                X_labeled=X_lab, y_labeled=y_lab,
                X_unlabeled=X_unl, y_unlabeled=y_unl,
                X_test=X_te, y_test=y_te,
            )
            history = semi_trainer.history
            logger.info(
                f"半监督训练完成，测试损失: {semi_metrics['test_loss']:.6f}"
            )

        save_artifacts(model, scaler)
        log_training_summary(history)
        run_prediction_example(raw_data, dates)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(config.file.figures_dir, f'training_history_{args.model}_{timestamp}.png')
        plot_training_history(history, save_path=plot_path)
        logger.info(f"训练历史图表已保存到: {plot_path}")

        logger.info("空气质量预测系统执行完成")
        return 0
    except KeyboardInterrupt:
        logger.warning("程序被用户中断")
        return 1
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        print(f"错误: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

