import os
import os.path
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision#这个库可以看作是pytorch在cv领域的拓展，有datasets，transforms，utils，models四个主要的类
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tensorboardX import SummaryWriter
from tqdm import tqdm#进度条



os.environ['CUDA_VISIBLE_DEVICES'] = "0"


data = glob.glob(r"Multi_data/image/*.png")
label = glob.glob(r"Multi_data/label/*.png")


category_map={0:1,80:2,160:3,255:0}


ratio = 0.8
train_lengh = int(ratio*len(data))

train_data = data[:train_lengh]#选取前800张为训练集
train_label = label[:train_lengh]
test_data = data[train_lengh:]
test_label = label[train_lengh:]
#trainsfrom
train_transform = transforms.Compose([
    transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST),#设置输入size
    transforms.RandomRotation(45),#随机旋转45度
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.NEAREST),   # 随机裁剪和缩放
    transforms.RandomHorizontalFlip(),   # 随机水平翻转 
    transforms.RandomVerticalFlip(),#随机垂直翻转
    # transforms.ToTensor(),  # 转换为张量 
])#训练集的输入格式

test_transform = transforms.Compose([
    transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.ToTensor()
])#验证集的输入格式
class BrainMRIdataset(Dataset):#继承Dataset，例如__getitem__()和__len__()是PyTorch数据加载器所必需的
    def __init__(self,img,mask,transformer):
        self.img=img
        self.mask=mask
        self.transformer=transformer
    
    def __getitem__(self,index):#获取数据集中指定索引的数据项，需要在自定义数据集类中实现
        img=self.img[index]# 获取图片路径，[index]操作符来获取数据集中指定索引的数据项，即获取第index个图片和掩膜路径
        mask=self.mask[index]# 获取mask路径
    
        random_int = torch.randint(1, 11, ())
        torch.manual_seed(random_int)#固定随机模式
        img_open=Image.open(img)#用PIL库中的Image.open()函数打开图片
        img_tensor=self.transformer(img_open)#转换
        img_tensor = transforms.ToTensor()(img_tensor)
        mask_open=Image.open(mask).convert('L')#mask.convert('L')
        mask_tensor=self.transformer(mask_open)
        mask_tensor = np.array(mask_tensor)
        mask_tensor = torch.from_numpy(mask_tensor)


        mask_tensor=torch.squeeze(mask_tensor).type(torch.long)#将掩膜tensor进行压缩，unsqueeze去掉维度为1的维度，并将其类型转换为long，而unsqueeze(0) 的作用是增加一个维度
        #数值转为类别
        for k, v in category_map.items():
            mask_tensor = torch.where(mask_tensor==k, v, mask_tensor)
        #torch.squeeze() 将所有维度为1的维度都删除,除那些不必要的、单纯增加维度数量的维度
        #使用 torch.long 目的就是将二值化的mask矩阵由浮点型或其它类型转为长整型(整型32位，长64位)
        #PyTorch 中的整数类型（包括 torch.int 和 torch.long）与浮点数类型（如 torch.float 和 torch.double）需要进行类型转换
        torch.seed()#取消固定随机
        return img_tensor,mask_tensor
    
    def __len__(self):#在这个方法中，代码只需要返回数据集中self.img 的长度即可，表示该数据集包含的总样本数量
        return len(self.img)
    
    
train = BrainMRIdataset(train_data,train_label,train_transform)#dataset
test = BrainMRIdataset(test_data,test_label,test_transform)    

dl_train = DataLoader(train,batch_size=32,shuffle=True)#dataloder输入训练集，batch_size=32 表示每次迭代返回 32 个样本组成的数据批次，shuffle=True 表示对每个 epoch 的训练数据先进行打乱再进行训练
dl_test = DataLoader(test,batch_size=32,shuffle=True)#dataloder输入测试集
#获取不同类别的训练权重
class_counts = torch.tensor([0,0,0,0])
for _,mask in dl_train:
    # 假设mask是numpy数组
    mask = mask.reshape(-1)
    class_counts += torch.bincount(mask)
class_weight = torch.sum(class_counts)/class_counts

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
        return(x5)

writer = SummaryWriter(log_dir='logs')#创建一个tensorboardx对象
model = Unet()#创建Unet对象

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)#载入cuda

loss_fn= nn.CrossEntropyLoss(weight = class_weight).cuda()#交叉熵损失函数，也可以用DiceLoss，一般二分类常用Diceloss
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=1e-6)#,weight_decay=0.000001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.9)
#定义梯度，model.parameters()表示将模型的可训练参数传递给优化器，以进行参数更新,梯度下降的方法为adam,weight_decay=1e-2表示正则化的系数为0.01

from tqdm import tqdm#进度条
def calculate_iou(pred, label, num_classes):
    ious = []
    B = pred.shape[0]
    pred = pred.reshape(B,-1)
    label = label.reshape(B,-1)

    for cls_name in range(1,num_classes):#0类为背景
        pred_inds = pred == cls_name
        label_inds = label == cls_name
        intersection = torch.logical_and(pred_inds, label_inds).sum(dim=1).float()  # 相交的数量
        union = torch.logical_or(pred_inds, label_inds).sum(dim=1).float()  #并集的数量
        iou = intersection / union
        iou = torch.where(torch.isnan(iou), torch.zeros_like(iou), iou)
        ious.append(iou)
    return torch.stack(ious,dim=1)



def fit(epoch, model, trainloader, testloader):#fit是一个封装模型训练步骤的方法
    # 将模型初始化为某个初始状态。
    # 循环遍历训练数据集的每个批次（batch），将输入数据传递给模型，并计算模型的预测输出。
    # 使用预测输出和真实标签（目标）之间的差异来计算损失值。
    # 根据损失值计算梯度，并使用优化算法（如随机梯度下降）来更新模型参数。
    # 重复步骤2-4，直到达到预定的迭代次数或损失函数收敛为止。

    #训练模式
    correct = 0#准确率初始值为0
    total = 0
    running_loss = 0
    epoch_iou = []
    model.train()#训练模式
    for x, y in tqdm(trainloader):#遍历trainloader,x为一个batch的image，y为一个批次的label
        x, y = x.to(device), y.to(device)
        y_pred = model(x)#第一个批次image传给Unet，得到输出
        loss = loss_fn(y_pred, y)#输出与label相比较计算loss值
        optimizer.zero_grad()#上一波的梯度清零
        loss.backward()#反向传播，loss值指导梯度下降
        optimizer.step()#梯度下降
        scheduler.step()#更新lr
        with torch.no_grad():#以下不进行梯度计算
            y_pred = torch.argmax(y_pred, dim=1)#找出最大值的索引，也就是预测分割出的那部分像素的索引值，dim=1也就是返回一个批次中每张图片的预测结果的那部分像素的索引，因为这是一个batch，所以dim=1才是每张图片的维度
            correct += (y_pred == y).sum().item()#将预测正确的样本数累加到correct变量中
            total += y.size(0)# 将当前批次的样本数量（即y的批次大小）累加到total变量中。
            running_loss += loss.item()#将当前批次的损失值（loss）以标量形式累加到running_loss变量中
            # intersection = torch.logical_and(y, y_pred)#交集
            # union = torch.logical_or(y, y_pred)#并集
            # batch_iou = torch.sum(intersection) / torch.sum(union)#交并比
            batch_iou = calculate_iou(y_pred,y,4)
            epoch_iou.append(batch_iou)#batch_iou由张量与张量计算得到的一个张量，item()可以把这个张量转为标量，注意只能转一个元素的张量,比如tensor([2])，而tensor([2,2])就会报错
    epoch_loss = running_loss / len(trainloader.dataset)#一个epoch中所有batch的loss累加的loss闭上总次数，得到这个epoch的平均loss
    #epoch_acc = correct / (total*256*256)#准确率

    #计算三类分割标签的平均iou
    train_iou = torch.round(torch.mean(torch.cat(epoch_iou))*1000)/1000#IOU

    #验证模式
    test_correct = 0
    test_total = 0
    test_running_loss = 0 
    epoch_test_iou = []
    model.eval()#验证模式
    with torch.no_grad():
        for x, y in tqdm(testloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)
            test_correct += (y_pred == y).sum().item()
            test_total += y.size(0)
            test_running_loss += loss.item()
            # intersection = torch.logical_and(y, y_pred)
            # union = torch.logical_or(y, y_pred)
            # batch_iou = torch.sum(intersection) / torch.sum(union)
            batch_iou = calculate_iou(y_pred,y,4)
            epoch_test_iou.append(batch_iou)
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    #epoch_test_acc = test_correct / (test_total*256*256)
    val_iou = torch.round(torch.mean(torch.cat(epoch_test_iou))*1000)/1000#IOU

    #这5段代码的作用是在训练过程中把模型保存到model文件中
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)#创建model目录，exist_ok=True如果目录已存在也不会引发错误
    static_dict = model.state_dict()#意思是将模型的参数（包括权重和偏置等）保存为一个字典对象，键是参数的名称，值是对应参数的数值
    model_path = f"model/epoch_{epoch}_train_mIou_{train_iou.item():.3f}_test_mIou_{val_iou.item():.3f}.pth"#round(x,3)对浮点数四舍五入保留三位小数，np.mean(epoch_iou)是取一个epoch中所有batch的iou平均值
    torch.save(static_dict, model_path)#保存为pth文件，也就是把static_dict保存到pth文件中
    
    print('epoch: ', epoch, 
          'loss: ', round(epoch_loss, 3),
          #'accuracy:', round(epoch_acc, 3),
          'IOU:', train_iou.item(),
          'val_loss: ', epoch_test_loss,
          #'test_accuracy:', round(epoch_test_acc, 3),
           'val_iou:', val_iou.item()
             )
        
    return epoch_loss, train_iou, epoch_test_loss, val_iou

epochs = 100
for epoch in range(epochs):#开始跑全部epochs
    epoch_loss, epoch_iou, epoch_test_loss, epoch_test_iou = fit(epoch,
                                                                 model,
                                                                 dl_train,
                                                                 dl_test)

    writer.add_scalar('Loss/train', epoch_loss, epoch)#把数据添加进tensorboardx
    writer.add_scalar('IOU/train', epoch_iou, epoch)
    writer.add_scalar('Loss/val', epoch_test_loss, epoch)
    writer.add_scalar('IOU/val', epoch_test_iou, epoch)
    writer.flush()#写入文件
writer.close()#关闭