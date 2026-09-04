from typing import Dict

from metric_matching.models.unet import UNetModel
from metric_matching.models.mlp import MLPEncoder


# helper factory for encoders
def build_encoder(model_cfg: Dict, default_image_size=(28, 28)):
    name = model_cfg.get("name").lower()
    params = model_cfg.get("params", {}) or {}
    if name == "unet":
        # allow config to override these; fall back to sensible defaults
        in_ch = params.get("in_channels", 1)
        image_size = params.get("image_size", default_image_size)
        model_channels = params.get("model_channels", 32)
        num_res_blocks = params.get("num_res_blocks", 1)
        rank = params.get("rank", 32)
        channel_factor = params.get("channel_factor", 1)
        attention_resolutions = params.get("attention_resolutions", (4,))
        channel_mult = params.get("channel_mult", (1, 2, 2))
        dropout = params.get("dropout", 0.0)
        num_heads = params.get("num_heads", 1)
        num_head_channels = params.get("num_head_channels", -1)
        out_bias = params.get("out_bias", False)
        return UNetModel(
            in_channels=in_ch,
            image_size=tuple(image_size),
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            out_channels=rank * channel_factor,
            attention_resolutions=tuple(attention_resolutions),
            dropout=dropout,
            channel_mult=tuple(channel_mult),
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            out_bias=out_bias,
        )
    elif name == "mlp":
        input_dim = params.get("input_dim", 2)
        hidden_dim = params.get("hidden_dim", 128)
        num_layers = params.get("num_layers", 2)
        # accept both canonical names and older aliases
        rank = params.get("rank", 2)
        out_dim = params.get("output_dim", 2)
        time_embedding = params.get("time_embedding", False)
        return MLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            rank=rank,
            out_dim=out_dim,
            time_embedding=time_embedding,
            temb_dim=params.get("temb_dim", None),
            dropout=params.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown encoder name: {name}")
