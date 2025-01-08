from UCTransNet.uc_transnet import UC_TransNet
from DCSAU_Net_main.pytorch_dcsaunet.DCSAU_Net import Model as DCSAU_Net
from ResU_Net.r2unet import ResU_Net
from UTNet.utnet import UTNet
from PDAttUnet.Architectures import PAttUNet
from TransAttUnet.TransAttUnet import UNet_Attention_Transformer_Multiscale as TransAttUnet
from UNet_2plus.UNet_2plus import UNet_2plus
from Attention_UNet.Attention_UNet import Att_UNet
from UNet.UNet import UNet

# EOF
#


# UNet UNet_2plus Att_UNet     UC_TransNet   ResU_Net   UTNet  PAttUNet   DCSAU_Net  TransAttUnet


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = DCSAU_Net(1).cuda()

    # net.load_from(weights=np.load(config_vit.pretrained_path))
    # model = ViT_seg(num_classes=1).to('cuda')

    flops, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=False)
    print('flops:', flops)
    print('params:', params)