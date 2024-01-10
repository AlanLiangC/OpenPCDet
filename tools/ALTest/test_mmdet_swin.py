import torch
from mmdet.models.backbones import SwinTransformer

backbone=dict(
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=False)

def test():
    model = SwinTransformer(**backbone)

    data = torch.randn(8,3,224,224)

    model = model.to('cuda')
    data = data.to('cuda')

    result = model(data)

    print(result.type())

if __name__ == "__main__":
    test()