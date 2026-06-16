"""
模型工厂函数
"""
from .base import BaseModel


def create_model(model_type: str, input_size: int, **kwargs) -> BaseModel:
    """创建指定类型的模型"""
    from .transformer import TransformerModel
    from .lstm import LSTMModel
    from .cnn import CNNModel
    from .hybrid import HybridModel

    model_classes = {
        'transformer': TransformerModel,
        'lstm': LSTMModel,
        'cnn': CNNModel,
        'hybrid': HybridModel,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"不支持的模型类型: {model_type}，可选: {list(model_classes.keys())}"
        )

    return model_classes[model_type](input_size=input_size, **kwargs)
