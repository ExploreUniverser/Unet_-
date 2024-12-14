import torch
from torchvision import transforms
from PIL import Image

import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Encoder,self).__init__()

        self.conv_relu=nn.Sequential( #nn.Sequential是一个有序的容器，作用是方便构建顺序模型，即按照一定的顺序将多个子模块组合成一个完整的模型
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),#nn.Conv2d二维图像卷积，kernel_size卷积核的大小3*3，padding代表卷积操作前在周围添加一圈值为零的一个像素的填充
            nn.ReLU(inplace=True),#nn.ReLU使用ReLU激活函数，inplace=True为一种内存优化技术
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2)#定义最大池化部分，stride卷积核每次移动两个像素，常用最大池化，其他的还有平均池化等

    def forward(self,x,if_pool=True):#控制函数，控制什么时候需要最大池化
        if if_pool:
            x = self.pool(x)#池化写在前面是为了组装的时候方便控制
        x = self.conv_relu(x)#执行卷积操作和激活函数操作
        return x

class Decoder(nn.Module):
    def __init__(self,channels):
        super(Decoder,self).__init__()
        self.conv_relu=nn.Sequential(##定义了一个由两个卷积层和 ReLU 激活函数构成的序列
            nn.Conv2d(2*channels,channels,kernel_size=3,padding=1),#2*channels下采样结果拿过来拼接
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.upconv_relu=nn.Sequential(#定义了一个反卷积层和 ReLU 激活函数组成的序列。反卷积层可以将编码器的低分辨率特征图上采样到原始大小。
            nn.ConvTranspose2d(channels,channels//2,kernel_size=3,stride=2,padding=1,output_padding=1),#反卷积层函数，用于执行上采样操作
            #拼接后，channels//2通道数除以2，stride=2步长
            nn.ReLU(inplace=True)
        )
    def forward(self,x):#前馈函数，反卷积操作
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.encode1=Encoder(3,64)#这里输入为3通道 
        self.encode2=Encoder(64,128)
        self.encode3=Encoder(128,256)
        self.encode4=Encoder(256,512)
        self.encode5=Encoder(512,1024)
        self.upconv_relu=nn.Sequential(
            nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=1,output_padding=1),#U最下面那层
            nn.ReLU(inplace=True)
        )  
        self.decode1=Decoder(512)
        self.decode2=Decoder(256)
        self.decode3=Decoder(128)#这里只到128，后续在下面设置

        self.convDouble = nn.Sequential(#这里是最后的那层
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.last=nn.Conv2d(64,4,kernel_size=1)#最终输出，64是in_channels，最后是几分类那么out_channels就是几，这里是2分类，最后一个卷积层是1*1的核

    def forward(self,x):#整个流程，x是每一层的输出
        x1=self.encode1(x,if_pool=False)
        x2=self.encode2(x1)
        x3=self.encode3(x2)
        x4=self.encode4(x3)
        x5=self.encode5(x4)

        x5=self.upconv_relu(x5)

        x5=torch.cat([x4,x5],dim=1)#torch.cat拼接操作，dim=1是一个batch([8, 2, 256, 256])的2这个元素，也就是最终像素分类数
        x5=self.decode1(x5)
        x5=torch.cat([x3,x5],dim=1)
        x5=self.decode2(x5)
        x5=torch.cat([x2,x5],dim=1)
        x5=self.decode3(x5)
        x5=torch.cat([x1,x5],dim=1)
        
        x5=self.convDouble(x5)
        x5=self.last(x5)
        if torch.any(x5[:,3,:,:]>0):
            pass
        return(x5)


# 加载模型
my_model = Unet()
state_dict = torch.load("model/epoch_99_train_mIou_0.10300000756978989_test_mIou_0.07000000029802322.pth")
my_model.load_state_dict(state_dict)
my_model = my_model.to("cuda")
my_model.eval()
transform = transforms.Compose([
    transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量 
])
category_map_re={1:0,2:80,3:160,0:255}
img_path = 'Multi_data/image/0001.png'
save_ask_path = './1.png'
image = Image.open(img_path)
input_image = transform(image).unsqueeze(0).to("cuda")
my_model.eval()
with torch.no_grad():
        output = my_model(input_image)
        segmentation_mask = torch.argmax(output, dim=1)
        for k, v in category_map_re.items():
            segmentation_mask = torch.where(segmentation_mask==k, v, segmentation_mask)
        segmentation_mask = segmentation_mask.to(torch.uint8).squeeze().cpu().numpy()
        segmentation_mask = Image.fromarray(segmentation_mask)
        segmentation_mask.save(save_ask_path)


# path_list = os.listdir("test")

# for i in path_list:
#     image = Image.open("test//"+i).convert("RGB")
#     input_image = transform(image).unsqueeze(0).to("cuda")
#     with torch.no_grad():
#         output = my_model(input_image)
#     threshold = 0.5
#     segmentation_mask = (output > threshold).squeeze().cpu().numpy()

#     image = Image.fromarray(segmentation_mask[1])

#     image.save("test2//"+i)