from dataclasses import dataclass


@dataclass
class BaseConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    attention_dropout: float = 0.0

@dataclass
class VisionConfig(BaseConfig):
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    use_head: bool = True


@dataclass
class TextConfig(BaseConfig):
    vocab_size: int = 32000
    max_embedding_positions: int = 64