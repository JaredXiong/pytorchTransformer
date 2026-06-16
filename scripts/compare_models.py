"""
模型对比脚本 - 训练并比较所有模型类型的性能

用法：
    python -m scripts.compare_models
"""
import argparse
import logging
import os
import sys
import time

import torch

# 让 `python -m scripts.compare_models` 能 import air_quality
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor, get_device, split_data
from air_quality.models import create_model
from air_quality.training import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_TYPES = ['transformer', 'lstm', 'cnn', 'hybrid']


def train_and_evaluate(model_type: str, train_loader, test_loader, device) -> dict:
    """训练并评估单个模型"""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"训练模型: {model_type}")
    logger.info(f"{'=' * 50}")

    model = create_model(model_type=model_type, input_size=config.model.input_size).to(device)
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        best_model_path=None,  # 不在对比过程中保存中间权重
        loss_type=config.training.loss_type,
        early_stop_patience=config.training.early_stop_patience,
        gradient_clip=config.training.gradient_clip,
        learning_rate=config.training.learning_rate,
    )

    start = time.time()
    trainer.train(num_epochs=config.training.epochs)
    elapsed = time.time() - start

    return {
        'model_type': model_type,
        'best_loss': trainer.best_loss,
        'best_epoch': trainer.history.get('best_epoch', 0) + 1,
        'epochs_trained': trainer.history.get('epochs_trained', 0),
        'training_time': elapsed,
        'train_loss_history': trainer.history.get('train_loss', []),
        'test_loss_history': trainer.history.get('test_loss', []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare all model variants')
    parser.add_argument(
        '--models', nargs='+', default=MODEL_TYPES,
        choices=['transformer', 'lstm', 'cnn', 'hybrid'],
        help='要比较的模型列表',
    )
    args = parser.parse_args()

    device = get_device()
    logger.info(f"使用设备: {device}")

    logger.info(f"正在加载数据: {config.data.data_file}")
    processor = AirQualityDataProcessor(
        seq_length=config.data.seq_length,
        prediction_days=config.data.prediction_days,
        rolling_windows=config.data.rolling_windows,
    )
    data, scaler, features, dates = processor.load_and_preprocess_data(config.data.data_file)
    X, y = processor.create_sequences(data)
    X_train, X_test, y_train, y_test = split_data(X, y, config.data.train_split_ratio)
    logger.info(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

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

    results = [train_and_evaluate(m, train_loader, test_loader, device) for m in args.models]

    logger.info("\n" + "=" * 80)
    logger.info("模型性能对比")
    logger.info("=" * 80)
    logger.info(
        f"{'模型类型':<15} {'最佳验证损失':<15} {'最佳轮次':<10} "
        f"{'训练轮次':<10} {'训练时间':<12}"
    )
    logger.info("-" * 80)
    for r in results:
        logger.info(
            f"{r['model_type']:<15} {r['best_loss']:<15.6f} "
            f"{r['best_epoch']:<10} {r['epochs_trained']:<10} {r['training_time']:<12.2f}s"
        )

    best = min(results, key=lambda x: x['best_loss'])
    logger.info(
        f"\n最佳模型: {best['model_type']}, 验证损失: {best['best_loss']:.6f}"
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
