import torch
import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):

    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):

    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):

    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1	#residual learning

class RCNN_block(nn.Module):

    def __init__(self,ch_in,ch_out,t=2):
        super(RCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x = self.RCNN(x)
        return x 
        
class ResCNN_block(nn.Module):

    def __init__(self,ch_in,ch_out):
        super(ResCNN_block,self).__init__()
        self.Conv = conv_block(ch_in, ch_out)
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x1 = self.Conv_1x1(x)
        x = self.Conv(x)
        return x+x1 

class U_Net(nn.Module):

    def __init__(self,num_classes=1, input_channels=3, deep_supervision=False):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=input_channels,ch_out=32)
        self.Conv2 = conv_block(ch_in=32,ch_out=64)
        self.Conv3 = conv_block(ch_in=64,ch_out=128)
        self.Conv4 = conv_block(ch_in=128,ch_out=256)
        self.Conv5 = conv_block(ch_in=256,ch_out=512)

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv5 = conv_block(ch_in=512, ch_out=256)

        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32,num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=input_channels,ch_out=32,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class RecU_Net(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,t=2):
        super(RecU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RCNN1 = RCNN_block(ch_in=input_channels,ch_out=32,t=t)

        self.RCNN2 = RCNN_block(ch_in=32,ch_out=64,t=t)
        
        self.RCNN3 = RCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RCNN4 = RCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RCNN5 = RCNN_block(ch_in=256,ch_out=512,t=t)
        

        self.Up5 = up_conv(ch_in=512,ch_out=256)
        self.Up_RCNN5 = RCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up4 = up_conv(ch_in=256,ch_out=128)
        self.Up_RCNN4 = RCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up3 = up_conv(ch_in=128,ch_out=64)
        self.Up_RCNN3 = RCNN_block(ch_in=128, ch_out=64,t=t)
        
        self.Up2 = up_conv(ch_in=64,ch_out=32)
        self.Up_RCNN2 = RCNN_block(ch_in=64, ch_out=32,t=t)

        self.Conv_1x1 = nn.Conv2d(32,num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ResU_Net(nn.Module):

    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,):
        super(ResU_Net,self).__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.ResCNN1 = ResCNN_block(ch_in=input_channels,ch_out=nb_filter[0])

        self.ResCNN2 = ResCNN_block(ch_in=nb_filter[0],ch_out=nb_filter[1])
        
        self.ResCNN3 = ResCNN_block(ch_in=nb_filter[1],ch_out=nb_filter[2])
        
        self.ResCNN4 = ResCNN_block(ch_in=nb_filter[2],ch_out=nb_filter[3])
        
        self.ResCNN5 = ResCNN_block(ch_in=nb_filter[3],ch_out=nb_filter[4])
        

        self.Up5 = up_conv(ch_in=nb_filter[4],ch_out=nb_filter[3])
        self.Up_ResCNN5 = ResCNN_block(ch_in=nb_filter[4], ch_out=nb_filter[3])
        
        self.Up4 = up_conv(ch_in=nb_filter[3],ch_out=nb_filter[2])
        self.Up_ResCNN4 = ResCNN_block(ch_in=nb_filter[3], ch_out=nb_filter[2])
        
        self.Up3 = up_conv(ch_in=nb_filter[2],ch_out=nb_filter[1])
        self.Up_ResCNN3 = ResCNN_block(ch_in=nb_filter[2], ch_out=nb_filter[1])
        
        self.Up2 = up_conv(ch_in=nb_filter[1],ch_out=nb_filter[0])
        self.Up_ResCNN2 = ResCNN_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0],num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.ResCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.ResCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.ResCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.ResCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.ResCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_ResCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_ResCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_ResCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_ResCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
