"""
训练脚本 - 端到端训练流程

用法：
    python -m scripts.train                        # 训练默认 Hybrid 模型
    python -m scripts.train --model lstm           # 训练指定模型
    python -m scripts.train --epochs 50            # 自定义训练轮数
"""
import argparse
import logging
import os
import sys
import time
from datetime import timedelta

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
from air_quality.data import AirQualityDataProcessor, split_data
from air_quality.models import create_model
from air_quality.training import ModelTrainer
from air_quality.inference import predict_air_quality, format_prediction_result

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


def train_model(model_type: str, train_loader, test_loader, device):
    """训练指定类型的模型"""
    logger.info(f"创建模型类型: {model_type}")
    model = create_model(
        model_type=model_type,
        input_size=config.model.input_size,
        output_size=config.model.output_size,
    ).to(device)

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
    )

    logger.info(f"开始训练 {model_type} 模型 ...")
    start = time.time()
    trainer.train(num_epochs=config.training.epochs)
    logger.info(f"训练完成，耗时: {time.time() - start:.2f} 秒")
    return model, trainer.history


def save_artifacts(model, scaler, model_type: str) -> None:
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


def run_prediction_example(raw_data, dates) -> None:
    """运行预测示例（生产模式：forecast）"""
    if len(raw_data) < config.data.seq_length:
        logger.warning(
            f"数据长度({len(raw_data)})不足以创建预测序列({config.data.seq_length})"
        )
        return

    logger.info("执行预测示例")
    sample_input = raw_data[-config.data.seq_length:]

    prediction_result = predict_air_quality(
        input_sequence=sample_input,
        num_days=config.data.prediction_days,
    )
    print(format_prediction_result(prediction_result, show_details=True))


def parse_args():
    parser = argparse.ArgumentParser(description='Air Quality Time-Series Training')
    parser.add_argument(
        '--model', default='hybrid',
        choices=['transformer', 'lstm', 'gru', 'cnn', 'tcn', 'hybrid'],
        help='要训练的模型类型 (默认: hybrid)',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        logger.info("空气质量预测系统启动")

        setup_directories()
        train_loader, test_loader, scaler, features, dates, data, raw_data, _ = prepare_data()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")

        model, history = train_model(args.model, train_loader, test_loader, device)
        save_artifacts(model, scaler, model_type=args.model)
        log_training_summary(history)
        run_prediction_example(raw_data, dates)

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
