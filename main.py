import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import math
import joblib
from datetime import timedelta
from typing import Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=4, dropout=0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.encoder = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # 标准Transformer编码器结构
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(d_model, input_size)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.decoder.bias.data.zero_()

    def forward(self, src):
        # 输入编码
        src = self.encoder(src) * math.sqrt(self.d_model)
        residual = src
        # 位置编码
        src = self.pos_encoder(src)
        src = self.norm1(src + residual)
        src = self.dropout(src)

        # Transformer编码器
        output = self.transformer_encoder(src)
        # 解码输出
        return self.decoder(output)


def load_and_preprocess_data(file_path) -> Tuple[np.ndarray, StandardScaler, list, pd.Series]:
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        logger.info(f"成功读取Excel文件: {file_path}，共 {len(df)} 行数据")
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}")
        raise

    # 日期处理
    if 'pubtime' not in df.columns:
        raise ValueError("Excel文件中未找到'pubtime'列")

    dates = df['pubtime'].copy()
    df['pubtime'] = pd.to_datetime(df['pubtime'], format='%Y/%m/%d', errors='coerce')
    if df['pubtime'].isnull().any():
        raise ValueError("pubtime列包含无效日期格式")

    # 提取时间特征（仅保留月份和季节）
    df['month'] = df['pubtime'].dt.month
    df['season'] = (df['pubtime'].dt.month % 12 + 3) // 3

    # 定义特征列（7个污染特征 + 2个时间特征）
    feature_columns = [
        'aqi', 'pm2_5_24h', 'pm10_24h', 'no2_24h', 'so2_24h', 'co_24h', 'o3_8h_24h',
        'month', 'season'
    ]

    # 列名映射
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().replace(' ', '')
        if col_lower == 'aqi':
            column_mapping[col] = 'aqi'
        elif 'pm2.5' in col_lower or 'pm25' in col_lower:
            column_mapping[col] = 'pm2_5_24h'
        elif 'pm10' in col_lower:
            column_mapping[col] = 'pm10_24h'
        elif 'no2' in col_lower:
            column_mapping[col] = 'no2_24h'
        elif 'so2' in col_lower:
            column_mapping[col] = 'so2_24h'
        elif 'co' in col_lower and 'code' not in col_lower:
            column_mapping[col] = 'co_24h'
        elif 'o3' in col_lower:
            column_mapping[col] = 'o3_8h_24h'

    df = df.rename(columns=column_mapping)
    available_features = [col for col in feature_columns if col in df.columns]

    # 验证特征数量（7+2=9）
    if len(available_features) != 9:
        missing = set(feature_columns) - set(available_features)
        logger.warning(f"特征缺失: {missing}")
        raise ValueError("特征数量不匹配")

    # 数据预处理
    data = df[available_features].values
    data = SimpleImputer(strategy='mean').fit_transform(data)
    scaler = StandardScaler().fit(data)

    return scaler.transform(data), scaler, available_features, dates


def create_sequences(data, seq_length, augmentation=True):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 3):  # 为预测未来3天留出空间
        x = data[i:i + seq_length]
        y = data[i + seq_length:i + seq_length + 3]  # 预测未来3个时间步

        if augmentation:
            # 改进的数据增强策略
            scale = np.random.normal(1.0, 0.02)
            noise = np.random.normal(0, 0.01, x.shape)  # 生成与x相同形状的噪声

            # 修正噪声形状匹配问题
            y_noise = np.random.normal(0, 0.01, y.shape)  # 为y生成独立噪声

            # 添加物理约束
            x = np.clip(x * scale + noise, a_min=0, a_max=None)
            y = np.clip(y * scale + y_noise, a_min=0, a_max=None)

        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    # 改进的损失函数（多任务加权）
    feature_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 0.5, 0.5])
    criterion = nn.SmoothL1Loss(reduction='none')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # 使用余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)

            # 多步预测损失
            loss = (criterion(output[:, -3:], batch_y) * feature_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证步骤
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = model(batch_x)
                test_loss += (criterion(output[:, -3:], batch_y) * feature_weights).mean().item()

        test_loss /= len(test_loader)
        scheduler.step()

        # 早停机制
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= 15:
                logger.info("提前停止训练")
                break
        
        if epoch % 10 == 0:
            logger.info(f"Epoch [{epoch}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss:.4f}")


def validate_prediction(prediction: np.ndarray) -> np.ndarray:
    """验证预测结果的物理合理性"""
    min_values = np.array([0, 0, 0, 0, 0, 0.3, 0])  # 各特征最小值
    max_ratios = np.array([1.0, 0.8, 0.7, 0.5, 0.1, 0.05, 1.0])  # 各污染物/AQI最大比例

    validated = np.clip(prediction, min_values, None)

    # 应用比例约束 (处理多天预测的情况)
    if len(validated.shape) == 2:  # 单天预测 shape: (n_samples, n_features)
        aqi = validated[:, 0]
        for i in range(1, 7):
            validated[:, i] = np.minimum(validated[:, i], aqi * max_ratios[i])
    elif len(validated.shape) == 3:  # 多天预测 shape: (n_samples, n_days, n_features)
        aqi = validated[:, :, 0]
        for i in range(1, 7):
            validated[:, :, i] = np.minimum(validated[:, :, i], aqi * max_ratios[i])

    return validated


def get_device():
    """检测并返回合适的设备(CUDA或CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")
    return device


def predict_air_quality(model_weights_path, scaler_path, input_sequence):
    device = get_device()
    scaler = joblib.load(scaler_path)

    model = TransformerModel(
        input_size=scaler.n_features_in_,
        d_model=128,
        nhead=4,
        num_layers=4
    ).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    input_scaled = scaler.transform(input_sequence)
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        # 取最后3个时间步作为未来3天的预测
        prediction = output[:, -3:, :].cpu().numpy()

    # 逆转换并验证
    prediction = scaler.inverse_transform(prediction.reshape(-1, prediction.shape[-1]))[:, :7]
    prediction = prediction.reshape(-1, 3, 7)  # 重新调整为 (batch, days, features)
    prediction = validate_prediction(prediction)

    # 添加季节调整因子（示例）
    season_factor = np.array([1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 1.1])  # 冬季调整
    prediction = prediction * season_factor

    return np.clip(prediction, a_min=0, a_max=500)


def main():
    # 配置参数
    EPOCHS = 100
    BATCH_SIZE = 32
    SEQ_LENGTH = 14
    LEARNING_RATE = 0.0008

    # 数据准备
    try:
        data, scaler, features, dates = load_and_preprocess_data('北京2015-2024.xlsx')
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return

    # 修改main函数中的数据处理流程
    X, y = create_sequences(data, SEQ_LENGTH)
    split = int(0.8 * len(X))
    
    # 正确划分训练集和测试集
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]

    # 仅使用训练集拟合标准化器
    scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # 初始化模型
    model = TransformerModel(
        input_size=len(features),
        d_model=128,
        nhead=4,
        num_layers=4
    )

    # 训练模型
    logger.info("开始训练模型...")
    train_model(model, train_loader, test_loader, EPOCHS, LEARNING_RATE)

    # 保存模型和标准化器
    torch.save(model.state_dict(), 'air_quality_model_weights.pth')
    joblib.dump(scaler, 'scaler.pkl')
    logger.info("模型和标准化器已保存")

    # 预测示例
    if len(data) >= SEQ_LENGTH:
        sample_input = data[-SEQ_LENGTH:]

        # 计算预测日期（最后日期+1天）
        last_date = pd.to_datetime(dates.iloc[-1])
        prediction_dates = [(last_date + timedelta(days=i+1)).strftime('%Y/%m/%d') for i in range(3)]

        # 执行预测
        predictions = predict_air_quality('best_model.pth', 'scaler.pkl', sample_input)

        # 输出结果
        print("\n未来3天空气质量预测:")
        print("=" * 50)
        pollution_features = features[:7]  # 只显示前7个污染特征
        for i in range(3):  # 3天预测
            print(f"\n预测日期: {prediction_dates[i]}")
            print("污染指标预测结果:")
            for feature, value in zip(pollution_features, predictions[0, i]):
                print(f"{feature}: {value:.2f}")


if __name__ == "__main__":
    main()