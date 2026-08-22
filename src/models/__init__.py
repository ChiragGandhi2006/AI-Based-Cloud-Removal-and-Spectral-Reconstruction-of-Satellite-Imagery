from .unet import UNet
from .attention_unet import AttentionUNet, AttentionBlock
from .encoders import Encoder, FeatureFusion, Decoder
from .attention import AttentionUNetFull
from .fusion import AttentionUNetFusion


def build_model(model_name="attention_unet", in_channels=13, out_channels=13,
                base_filters=32, **kwargs):
    """Factory helper to construct a model from a config-style name."""
    name = (model_name or "attention_unet").lower().replace("-", "_")

    if name in ("unet", "baseline"):
        return UNet(in_channels=in_channels, out_channels=out_channels,
                    base_filters=base_filters)
    if name in ("attention_unet", "attentionunet"):
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels,
                             base_filters=base_filters, **kwargs)
    if name in ("attention_unet_full", "attentionunetfull"):
        return AttentionUNetFull(in_channels=in_channels, out_channels=out_channels,
                                 base_filters=base_filters, **kwargs)
    if name in ("attention_unet_fusion", "attentionunetfusion", "sar_fusion"):
        return AttentionUNetFusion(in_channels=in_channels, out_channels=out_channels,
                                   base_filters=base_filters)
    raise ValueError(f"Unknown model name: {model_name!r}")