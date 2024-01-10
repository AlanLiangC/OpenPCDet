import torch
from pcdet.models.backbones_image.swin import WindowMSA



def test():
    model  = WindowMSA(embed_dims=256, num_heads=8,window_size=(5,5))
    data = torch.randn(256*4, 25, 256)

    model = model.to('cuda')
    data = data.cuda()

    result = model(data)

    print(result.shape)


if __name__ ==  "__main__":
    test()