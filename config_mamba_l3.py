from dataclasses import dataclass, field


@dataclass
class MambaConfigl3:

    d_model: int = 16
    d_intermediate: int = 16
    n_layer: int = 8
    vocab_size: int = 50277
    ssm_cfg: dict = None
    attn_layer_idx: list = None
    attn_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True