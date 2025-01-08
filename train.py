import argparse
import os
import torch
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

import archs

from DDS_UNet import DDS_UNet  # Ours

device = torch.device("cuda")

model_tab = [
"DDS_UNet",  # Ours
# "Att_UNet",
# "ResU_Net",
# "UTNet",
# "UC_TransNet",
# "PAttUNet",
# "DCSAU_Net",
# "TransAttUnet",

    ]
for i in  model_tab:
    arg_list = [200, 4, 256]
    rand_list = [41, 30, 23, 58]

    resume = 'Fault'
    # resume = 'True'
    val_num =1


    #####################################################################################
    models_name = i    #   UNet UNet_2plus  Att_UNet     UC_TransNet   ResU_Net   UTNet  PAttUNet  TransAttUnet DCSAU_Net  UIUNET SSTrans_Net ---------   OUNet
    dataset_name = "Dataset B"    #    "CVC-ClinicDB"       "buli"  "isic2018" "Kvasir-SEG"  "Dataset B"  "busi
    #####################################################################################

    if dataset_name == 'CVC-ClinicDB' :
        arg_list = [100, 8, 384]    # [epoch  batch_size image_size]
        val_k = 3

    if dataset_name == 'buli' :
        arg_list = [100, 8, 384]
        val_k = 3

    if dataset_name == 'Dataset B':
        arg_list = [100, 8, 384]
        val_k = 3

    if dataset_name == 'isic2018' :
        arg_list = [100, 8, 384]
        val_k = 3

    if dataset_name == 'Kvasir-SEG':
        arg_list = [100, 8, 384]
        val_k = 3

    if models_name == 'DDS_UNet':
        archs_name = DDS_UNet
    else:
        archs_name = archs

    models_num = 'models'

    def read_txt_file(file_path):

        with open(file_path, 'r') as file:
            content = file.read()
            result111 = []
            result = content.split("\n")
            for i in range(len(result)):
                if dataset_name == 'Glas' or dataset_name == 'PH2':
                    result111.append(result[i].split('/')[-1].split('.')[0])
                else:
                    result111.append(result[i].split('\\')[-1].split('.')[0])
            return result111

    def parse_args():
        parser = argparse.ArgumentParser()
        dataset_name_replace = dataset_name
        if dataset_name == 'isic2018':
            dataset_name_replace = 'isic'
        parser.add_argument('--name', default='{}_{}{}'.format(models_name,dataset_name_replace, val_num),
                            help='model name: (default: arch+timestamp)')
        parser.add_argument('--epochs', default=arg_list[0], type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('-b', '--batch_size', default=arg_list[1], type=int,
                            metavar='N', help='mini-batch size (default: 16)')

        # model
        models_name_replace = models_name
        parser.add_argument('--val_num', default=val_num, type=int)
        parser.add_argument('--arch', '-a', metavar='ARCH', default=models_name_replace)
        parser.add_argument('--deep_supervision', default=False, type=str2bool)
        parser.add_argument('--input_channels', default=3, type=int,
                            help='input channels')
        parser.add_argument('--num_classes', default=1, type=int,
                            help='number of classes')
        parser.add_argument('--input_w', default=arg_list[2], type=int,
                            help='image width')
        parser.add_argument('--input_h', default=arg_list[2], type=int,
                            help='image height')

        # loss
        parser.add_argument('--loss', default='BCEDiceLoss')

        # dataset
        parser.add_argument('--dataset', default=dataset_name,
                            help='dataset name')
        if dataset_name == 'busi'or dataset_name == 'CVC-ClinicDB' or dataset_name == 'Dataset B' :
            parser.add_argument('--img_ext', default='.png',
                                help='image file extension')

        if dataset_name == 'Glas' or dataset_name == 'PH2':
            parser.add_argument('--img_ext', default='.bmp',
                                help='image file extension')
        if dataset_name == 'buli':

            parser.add_argument('--img_ext', default='.bmp',
                                help='image file extension')

        if dataset_name == 'isic2018' or dataset_name == 'Kvasir-sessile' or dataset_name == 'Kvasir-SEG' or dataset_name == 'IDRiD':
            parser.add_argument('--img_ext', default='.jpg',
                                help='image file extension')

        if dataset_name == 'busi' or dataset_name == 'isic2018' or dataset_name == 'Dataset B' or dataset_name == 'covid19' or dataset_name == 'CVC-ClinicDB' or dataset_name == 'TNBC':
            parser.add_argument('--mask_ext', default='.png',
                                help='mask file extension')
        if dataset_name == 'buli':
            parser.add_argument('--mask_ext', default='.bmp',
                                help='mask file extension')
        if dataset_name == 'Kvasir-SEG':
            parser.add_argument('--mask_ext', default='.jpg',
                                help='mask file extension')
        if dataset_name == 'IDRiD':
            parser.add_argument('--mask_ext', default='.tif',
                                help='mask file extension')
        if dataset_name == 'STARE':
            parser.add_argument('--mask_ext', default='.ppm',
                                help='mask file extension')
        if dataset_name == 'Glas' or dataset_name == 'PH2':
            parser.add_argument('--mask_ext', default='.bmp',
                                help='mask file extension')
        # optimizer
        parser.add_argument('--optimizer', default='Adam',
                            choices=['Adam', 'SGD'],
                            help='loss: ' +
                                 ' | '.join(['Adam', 'SGD']) +
                                 ' (default: Adam)')

        parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float,
                            help='momentum')
        parser.add_argument('--weight_decay', default=1e-4, type=float,
                            help='weight decay')
        parser.add_argument('--nesterov', default=False, type=str2bool,
                            help='nesterov')

        # scheduler
        parser.add_argument('--scheduler', default='CosineAnnealingLR',
                            choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'StepLR', 'ConstantLR'])
        parser.add_argument('--min_lr', default=1e-5, type=float,
                            help='minimum learning rate')
        parser.add_argument('--factor', default=0.1, type=float)
        parser.add_argument('--patience', default=2, type=int)
        parser.add_argument('--milestones', default='1,2', type=str)
        parser.add_argument('--gamma', default=2 / 3, type=float)
        parser.add_argument('--early_stopping', default=-1, type=int,
                            metavar='N', help='early stopping (default: -1)')
        parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

        parser.add_argument('--num_workers', default=0, type=int)

        config = parser.parse_args()

        return config

    def train(config, train_loader, model, criterion, optimizer):
        avg_meters = {'loss': AverageMeter(),
                      'dice': AverageMeter()}

        model.train()

        pbar = tqdm(total=len(train_loader))
        for input, target, _ in train_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice = iou_score(output, target)

            # compute gradient and do optimizing step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('dice', avg_meters['dice'].avg)])


    def validate(config, val_loader, model, criterion):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'dice': AverageMeter()}

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            pbar = tqdm(total=len(val_loader))
            for input, target, _ in val_loader:
                input = input.to(device)
                target = target.to(device)

                # compute output
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, dice = iou_score(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    # iou, dice = iou_score(output[0], target)
                    iou, dice = iou_score(output, target)

                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                avg_meters['dice'].update(dice, input.size(0))

                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                    ('dice', avg_meters['dice'].avg)
                ])
                pbar.set_postfix(postfix)
                pbar.update(1)
            pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg)])


    def main():
        config = vars(parse_args())

        if config['name'] is None:
            if config['deep_supervision']:
                config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
            else:
                config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

        os.makedirs('%s/%s' % (models_num, config['dataset']), exist_ok=True)

        if os.path.exists('%s/%s/%s' % (models_num, config['dataset'],config['name'])):
            input('已有文件存在，是否继续覆盖')
        os.makedirs('%s/%s/%s' % (models_num, config['dataset'],config['name']), exist_ok=True)
        print('-' * 20)
        for key in config:
            print('%s: %s' % (key, config[key]))
        print('-' * 20)

        with open('%s/%s/%s/config.yml' % (models_num, config['dataset'], config['name']), 'w') as f:
            yaml.dump(config, f)

        # define loss function (criterion)
        if config['loss'] == 'BCEDiceLoss':
            criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            criterion = losses.__dict__[config['loss']]().to(device)

        cudnn.benchmark = True

        # create model
        model = archs_name.__dict__[config['arch']](config['num_classes'],
                                                    config['input_channels'],
                                                    config['deep_supervision']
                                                    )

        model = model.to(device)

        params = filter(lambda p: p.requires_grad, model.parameters())
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                params, lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                                  nesterov=config['nesterov'], weight_decay=config['weight_decay'])
        else:
            raise NotImplementedError

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                       verbose=1, min_lr=config['min_lr'])
        elif config['scheduler'] == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                                 gamma=config['gamma'])
        elif config['scheduler'] == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)
        else:
            raise NotImplementedError

        best_dice = 0
        now_epoch = 0
        log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('dice', []),
            ('val_loss', []),
            ('val_iou', []),
            ('val_dice', []),
        ])


        if resume == 'True':
            model.load_state_dict(torch.load('%s/%s/%s/last_model.pth' %
                           (models_num, config['dataset'], config['name'])))
            optimizer.load_state_dict(torch.load("%s/%s/%s/optim.pth"%
                       (models_num, config['dataset'], config['name'])))
            scheduler.load_state_dict(torch.load("%s/%s/%s/lr.pth"%
                       (models_num, config['dataset'], config['name'])))
            txt = open('%s/%s/%s/best_dice.txt' %
                       (models_num, config['dataset'], config['name']), "r")
            best_dice = float(txt.read())
            txt.close()
            txt2 = open('%s/%s/%s/epoch.txt' %
                       (models_num, config['dataset'], config['name']), "r")
            now_epoch = int(txt2.read())
            txt2.close()

            import csv
            with open('%s/%s/%s/log.csv'%
                       (models_num, config['dataset'], config['name'])) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    log['epoch'].append(row['epoch'])
                    log['lr'].append(row['lr'])
                    log['loss'].append(row['loss'])
                    log['dice'].append(row['dice'])
                    log['val_loss'].append(row['val_loss'])
                    log['val_iou'].append(row['val_iou'])
                    log['val_dice'].append(row['val_dice'])

        # Data loading code
        img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]


        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=rand_list[val_num - 1])

        print(train_img_ids)
        print(val_img_ids)

        print(len(train_img_ids),len(val_img_ids))
        train_transform = Compose([
            RandomRotate90(),
            transforms.Flip(),
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])

        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ])

        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join('inputs', config['dataset'], 'images'),
            mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=train_transform)
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join('inputs', config['dataset'], 'images'),
            mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            transform=val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)

        trigger = 0

        for epoch in range(now_epoch , config['epochs']):
            print('Epoch [%d/%d]' % (epoch, config['epochs']))
            # print(scheduler.state_dict()['_last_lr'])
            # train for one epoch
            train_log = train(config, train_loader, model, criterion, optimizer)
            # evaluate on validation set
            val_log = validate(config, val_loader, model, criterion)


            log['epoch'].append(epoch)
            log['lr'].append(scheduler.state_dict()['_last_lr'])
            log['loss'].append(train_log['loss'])
            log['dice'].append(train_log['dice'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])

            pd.DataFrame(log).to_csv('%s/%s/%s/log.csv' % (models_num, config['dataset'], config['name']), index=False)

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            if config['scheduler'] == 'StepLR':
                scheduler.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])

            print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
                  % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

            trigger += 1
            print(trigger)
            if val_log['dice'] > best_dice:
                torch.save(model.state_dict(), '%s/%s/%s/model.pth' %
                           (models_num, config['dataset'],config['name']))
                best_dice = val_log['dice']

                if best_dice >0.01:
                    txt = open('%s/%s/%s/best_dice.txt' %
                               (models_num, config['dataset'], config['name']), "w")
                    txt.write(str(best_dice))

                print("=> saved best model")
                trigger = 0

            torch.save(model.state_dict(), '%s/%s/%s/last_model.pth' %
                       (models_num, config['dataset'], config['name']))
            torch.save(optimizer.state_dict(), "%s/%s/%s/optim.pth" %
                       (models_num, config['dataset'], config['name']))
            torch.save(scheduler.state_dict(), "%s/%s/%s/lr.pth" %
                       (models_num, config['dataset'], config['name']))
            txt = open('%s/%s/%s/epoch.txt' %
                       (models_num, config['dataset'], config['name']), "w")
            txt.write(str(epoch+1))
            txt.close()
            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break
            torch.cuda.empty_cache()

    if __name__ == '__main__':
        main()