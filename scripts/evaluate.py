"""
回测评估脚本 - 对训练好的模型在完整数据集上做滚动预测评估。

用法：
    python -m scripts.evaluate
    python -m scripts.evaluate --model lstm
    python -m scripts.evaluate --days 7
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# 让 `python -m scripts.evaluate` 能 import air_quality
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from air_quality.config import config
from air_quality.data import AirQualityDataProcessor
from air_quality.inference import AirQualityPredictor
from air_quality.visualization import plot_backtest_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Backtest a trained model')
    parser.add_argument('--model', default='hybrid',
                        choices=['transformer', 'lstm', 'gru', 'cnn', 'tcn', 'hybrid'],
                        help='模型类型 (默认 hybrid)')
    parser.add_argument('--days', type=int, default=config.data.prediction_days,
                        help=f'预测天数 (默认 {config.data.prediction_days})')
    parser.add_argument('--data-file', default=config.data.data_file,
                        help=f'数据文件路径 (默认 {config.data.data_file})')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        logger.info(f"回测评估 - 模型: {args.model}, 预测天数: {args.days}")

        # 加载数据
        processor = AirQualityDataProcessor(
            seq_length=config.data.seq_length,
            prediction_days=args.days,
            rolling_windows=config.data.rolling_windows,
            output_size=config.model.output_size,
        )
        data, scaler, features, dates = processor.load_and_preprocess_data(args.data_file)
        logger.info(f"数据加载完成: {len(data)} 天, {len(features)} 污染物")

        # 重新加载原始未裁剪数据，用于图表的"真实值"曲线
        # 否则 _handle_outliers 会将 AQI 裁剪到 ≤500、CO 裁剪到 ≤50，
        # 导致图表 y 轴被强行压缩，无法反映真实数据范围。
        raw_df = pd.read_excel(args.data_file, engine='openpyxl')
        column_mapping = processor._create_column_mapping(raw_df.columns)
        raw_df = raw_df.rename(columns=column_mapping)
        # 按 processor 实际使用的 7 个污染物顺序提取原始数据
        available_features = [c for c in features if c in raw_df.columns]
        raw_pollutants = raw_df[available_features].values
        if raw_pollutants.shape[1] < 7:
            logger.warning(
                f"原始数据仅找到 {raw_pollutants.shape[1]} 个污染物列，"
                f"图表将使用预处理后的真值（可能被裁剪）"
            )
            y_true_raw = None
        else:
            # 对齐窗口：每个窗口预测的目标是 data[start+seq_len : start+seq_len+prediction_days]
            seq_len = config.data.seq_length
            n_samples = len(data) - seq_len - args.days + 1
            y_true_raw = np.array([
                raw_pollutants[start + seq_len:start + seq_len + args.days]
                for start in range(n_samples)
            ], dtype=float)

        # 加载预测器
        predictor = AirQualityPredictor()
        predictor.load_model()

        # 回测
        logger.info("开始回测 ...")
        start = time.time()
        result = predictor.backtest(data=data, dates=dates, prediction_days=args.days)
        elapsed = time.time() - start
        logger.info(f"回测完成，耗时 {elapsed:.2f} 秒, 共 {len(result['y_pred'])} 个窗口")

        # 打印指标
        metrics = result['metrics']
        logger.info("\n整体指标:")
        for k in ['overall_mse', 'overall_rmse', 'overall_mae', 'overall_r2']:
            logger.info(f"  {k}: {metrics[k]:.4f}")

        logger.info("\n分特征指标 (RMSE):")
        for feature in ['aqi', 'pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']:
            logger.info(f"  {feature}: {metrics[f'{feature}_rmse']:.4f}")

        # 画图：每个指标独立图表，保存到 outputs/figures/<model_type>/
        # y_true_raw 是原始未裁剪的真值，让 y 轴反映真实数据范围
        out_dir = plot_backtest_results(
            y_true=result['y_true'],
            y_pred=result['y_pred'],
            dates=result['dates'],
            model_type=args.model,
            y_true_raw=y_true_raw,
        )
        logger.info(f"图表已保存到: {out_dir}/")
        return 0
    except Exception as e:
        logger.error(f"回测失败: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
