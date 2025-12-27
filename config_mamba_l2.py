from dataclasses import dataclass, field


@dataclass
class MambaConfigl2:

    d_model: int = 32
    d_intermediate: int = 32
    n_layer: int = 16
    vocab_size: int = 50277
    ssm_cfg: dict = None
    attn_layer_idx: list = None
    attn_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True