import argparse
import os
from glob import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import archs

from DDS_UNet import DDS_UNet
from dataset import Dataset
from metrics import iou_score, all_score
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
val_k = 5

arg_list = [100, 8, 384]
rand_list = [41, 30, 23, 58]

model_tab = [
    "DDS_UNet"

    ]
# for i in  model_tab:
for i in range (1):



   resume = 'False'   #是否继续上一次中断的代码
   val_num = 1   #数据集划分随机数种子，暂时不要修改

#####################################################################################
models_name = "DDS_UNet"  #   UNet UNet_2plus Att_UNet     UC_TransNet   ResU_Net   UTNet  PAttUNet   DCSAU_Net  TransAttUnet  ---------   OUNet
dataset_name = "Dataset B"    #"CVC-ClinicDB" "isic2018" "buli"  "Dataset B"   Dataset_B_90  busi
#####################################################################################





if dataset_name == 'CVC-ClinicDB':
    arg_list = [100, 8, 384]    # [epoch  batch_size image_size]
    val_k = 3


if dataset_name == 'Dataset B':
    arg_list = [100, 8, 384]
    val_k = 3

if dataset_name == 'busi':
    arg_list = [100, 8, 384]
    val_k = 3

if dataset_name == 'buli':
    arg_list = [100, 8, 384]
    val_k = 3

if models_name == 'DDS_UNet':
    archs_name = DDS_UNet
else:
    archs_name = archs



models_num = 'models{}'.format(val_num)


def parse_args():
    parser = argparse.ArgumentParser()

    dataset_name____ = dataset_name
    if dataset_name == 'isic2018':
        dataset_name____ = 'isic'
    parser.add_argument('--name', default='{}_{}{}_1'.format(models_name, dataset_name____, val_num),
                        help='model name: (default: arch+timestamp)')

    parser.add_argument('--dataset', default=dataset_name,
                        help='model name')
    parser.add_argument('--models', default='models{}'.format(val_num),
                        help='model name')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    with open('%s/%s/%s/config.yml' % (args.models, args.dataset,args.name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    model = archs_name.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])



    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))       #config['img_ext']
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=rand_list[val_num - 1])

    print(val_img_ids)
    model.load_state_dict(torch.load('%s/%s/%s/model.pth' %
                                     (args.models, args.dataset, config['name'])))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
        # RandomRotate90(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['img_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    N=config['num_classes']

    iou_avg_meter = []
    dice_avg_meter = []
    acc_avg_meter = []
    pr_avg_meter = []
    re_avg_meter = []
    p_avg_meter = []
    for i in range(N):
        iou_avg_meter.append(AverageMeter())
        dice_avg_meter.append(AverageMeter())
        acc_avg_meter.append(AverageMeter())
        pr_avg_meter.append(AverageMeter())
        re_avg_meter.append( AverageMeter())
        p_avg_meter.append( AverageMeter())



    gput = AverageMeter()
    cput = AverageMeter()

    count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output= model(input)



            iou,dice,acc,pr,re,p = all_score(output, target)
            # print(dice)
            # hd95 = hd_score(output, target)
            for i in range(N):
                iou_avg_meter[i].update(iou[i], input.size(0))
                dice_avg_meter[i].update(dice[i], input.size(0))
                acc_avg_meter[i].update(acc[i], input.size(0))
                pr_avg_meter[i].update(pr[i], input.size(0))
                re_avg_meter[i].update(re[i], input.size(0))
                p_avg_meter[i].update(p[i], input.size(0))



            import numpy as np
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    image_array = np.transpose(output[i].cpu().numpy(), (1, 2, 0))
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                image_array*255)

    for i in range(N):
        print('IoU%d: %.4f'%(i, iou_avg_meter[i].avg))
        print('Dice%d: %.4f'%( i,dice_avg_meter[i].avg))
        print('Acc%d: %.4f'%(i,acc_avg_meter[i].avg))
        print('Precision%d: %.4f'%(i,pr_avg_meter[i].avg))
        print('Recall%d: %.4f'%(i,re_avg_meter[i].avg))
        print('p_value%d: %.4f'%(i,p_avg_meter[i].avg))
        print("-------------------------------------")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()