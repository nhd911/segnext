from lib.modules.encoder_decoder import EncoderDecoder
from lib.modules.ham_head import LightHamHead
from lib.modules.mscan import MSCAN


def create_segnext_t(num_classes=150):
    return EncoderDecoder(
        backbone=MSCAN(
            embed_dims=[32, 64, 160, 256],
            mlp_ratios = [8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            norm_cfg=dict(type='BN', requires_grad=True)
        ),
        decode_head=LightHamHead(
            in_channels=[64, 160, 256],
            in_index=[1, 2, 3],
            channels=256,
            ham_channels=256,
            dropout_ratio=0.1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_classes=num_classes
        )
    )


def create_segnext_s(num_classes):
    return EncoderDecoder(
        backbone=MSCAN(
            embed_dims=[64, 128, 320, 512],
            mlp_ratios = [8, 8, 4, 4],
            depths=[2, 2, 4, 2],
            norm_cfg=dict(type='BN', requires_grad=True)
        ),
        decode_head=LightHamHead(
            in_channels=[128, 320, 512],
            in_index=[1, 2, 3],
            channels=256,
            ham_channels=256,
            dropout_ratio=0.1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_classes=num_classes
        )
    )

def create_segnext_b(num_classes):
    return EncoderDecoder(
        backbone=MSCAN(
            embed_dims=[64, 128, 320, 512],
            mlp_ratios = [8, 8, 4, 4],
            depths=[3, 3, 12, 3],
            norm_cfg=dict(type='BN', requires_grad=True),
            drop_rate=0.1
        ),
        decode_head=LightHamHead(
            in_channels=[128, 320, 512],
            in_index=[1, 2, 3],
            channels=512,
            ham_channels=512,
            dropout_ratio=0.1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_classes=num_classes
        )
    )


def create_segnext_l(num_classes):
    return EncoderDecoder(
        backbone=MSCAN(
            embed_dims=[64, 128, 320, 512],
            mlp_ratios = [8, 8, 4, 4],
            depths=[3, 5, 27, 3],
            norm_cfg=dict(type='BN', requires_grad=True),
            drop_rate=0.3
        ),
        decode_head=LightHamHead(
            in_channels=[128, 320, 512],
            in_index=[1, 2, 3],
            channels=1024,
            ham_channels=1024,
            dropout_ratio=0.1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_classes=num_classes
        )
    )