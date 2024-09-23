import os
import argparse  # 用于解析命令行参数
import torch
import torch.optim as optim  # PyTorch中的优化器
from torch.utils.data import DataLoader  # PyTorch中用于加载数据的工具
from tqdm import tqdm  # 用于在循环中显示进度条
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
import torch.nn.functional as F  # PyTorch中的函数库
from torchvision import datasets  # PyTorch中的视觉数据集
import torchvision.transforms as transforms  # PyTorch中的数据变换操作
from tensorboardX import SummaryWriter  # 用于创建TensorBoard日志的工具
from utils import AverageMeter, accuracy  # 自定义工具模块，用于计算模型的平均值和准确度
from model import model_dict  # 自定义模型字典，包含了各种模型的定义
import numpy as np
import time
import random

# 设置环境变量以避免 OpenMP 错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_names", type=str, default="resnet18")
parser.add_argument("--pre_trained", type=bool, default=False)
parser.add_argument("--classes_num", type=int, default=30) #样品种类数
parser.add_argument("--dataset", type=str, default=r"D:\HZAU\exp\EEM\CNN\dataset\seed water")  #文件路径
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epoch", type=int, default=20) # 训练轮次
parser.add_argument("--lr", type=float, default=0.01) # 学习率，默认为0.01
parser.add_argument("--momentum", type=float, default=0.9) # 优化器的动量，默认为 0.9
parser.add_argument("--weight-decay", type=float, default=1e-4)  # 权重衰减（正则化项），默认为 5e-4
parser.add_argument("--seed", type=int, default=33) # 随机种子
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--print_freq", type=int, default=1) # 打印训练信息的频率，默认为 1（每个轮次打印一次）
parser.add_argument("--exp_postfix", type=str, default="seed33") ## 实验结果文件夹的后缀，默认为 "seed33"
parser.add_argument("--txt_name", type=str, default="lr0.01_wd5e-4")
args = parser.parse_args()


# 设置随机种子
def seed_torch(seed=74):
    random.seed(seed)  # 设置Python随机模块的种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个GPU，为所有GPU设置随机种子


seed_torch(seed=args.seed)  # 设置随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # 设置使用的GPU设备ID

# 构建实验结果文件夹的路径
exp_name = args.exp_postfix
base_path = r"D:\HZAU\exp\EEM\CNN\report"
exp_path = os.path.join(base_path, args.dataset, args.model_names, exp_name)

# 数据增强与转换
transform_train = transforms.Compose([
    transforms.RandomRotation(90),  # 随机旋转90度
    transforms.Resize([256, 256]),  # 将图片大小调整为256x256
    transforms.RandomCrop(224),  # 随机裁剪成224x224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),  # 转换成张量
    transforms.Normalize((0.3738, 0.3738, 0.3738), (0.3240, 0.3240, 0.3240))  # 标准化
])
transform_test = transforms.Compose([
    transforms.Resize([224, 224]),  # 将图片大小调整为224x224
    transforms.ToTensor(),  # 转换成张量
    transforms.Normalize((0.3738, 0.3738, 0.3738), (0.3240, 0.3240, 0.3240))  # 标准化
])

# 加载数据集
trainset = datasets.ImageFolder(root=os.path.join(args.dataset, 'train'), transform=transform_train)
testset = datasets.ImageFolder(root=os.path.join(args.dataset, 'test'), transform=transform_test) # test文件名内每类只含有一个数据，test-2内每类含有2个数据，使用3：1的数据分布，故使用test作为测试集
train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True)
test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True)

# 定义训练一个epoch的函数
train_losses_list = []  # 用于记录每轮训练的损失
train_acces_list = []  # 用于记录每轮训练的准确率
test_losses_list = []  # 用于记录每轮测试的损失
test_acces_list = []  # 用于记录每轮测试的准确率


def train_one_epoch(model, optimizer, train_loader):
    model.train()  # 设置模型为训练模式
    acc_recorder = AverageMeter()  # 用于记录准确率
    loss_recorder = AverageMeter()  # 用于记录损失

    for (inputs, targets) in tqdm(train_loader, desc="train"):  # 进度条显示训练状态
        if torch.cuda.is_available():  # 如果有可用的GPU，则将输入和目标转移到GPU上
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        out = model(inputs)  # 前向传播
        loss = F.cross_entropy(out, targets)  # 计算交叉熵损失
        loss_recorder.update(loss.item(), n=inputs.size(0))  # 更新损失记录器
        acc = accuracy(out, targets)[0]  # 计算准确率
        acc_recorder.update(acc.item(), n=inputs.size(0))  # 更新准确率记录器

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

    losses = loss_recorder.avg  # 获取平均损失
    acces = acc_recorder.avg  # 获取平均准确率
    return losses, acces  # 返回损失和准确率


# 定义评估函数
def evaluation(model, test_loader):
    model.eval()  # 设置模型为评估模式
    acc_recorder = AverageMeter()  # 用于记录准确率
    loss_recorder = AverageMeter()  # 用于记录损失

    with torch.no_grad():  # 不计算梯度
        for (inputs, targets) in tqdm(test_loader, desc="test"):  # 进度条显示测试状态
            if torch.cuda.is_available():  # 如果有可用的GPU，则将输入和目标转移到GPU上
                inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

            out = model(inputs)  # 前向传播
            loss = F.cross_entropy(out, targets)  # 计算交叉熵损失
            loss_recorder.update(loss.item(), n=inputs.size(0))  # 更新损失记录器
            acc = accuracy(out, targets)[0]  # 计算准确率
            acc_recorder.update(acc.item(), n=inputs.size(0))  # 更新准确率记录器

    losses = loss_recorder.avg  # 获取平均损失
    acces = acc_recorder.avg  # 获取平均准确率
    return losses, acces  # 返回损失和准确率


# 主训练函数
def train(model, optimizer, train_loader, test_loader, scheduler):
    since = time.time()  # 记录开始时间
    best_acc = -1  # 最佳准确率初始化为负无穷

    # 打开文件以记录训练过程
    f = open(os.path.join(exp_path, "{}.txt".format(args.txt_name)), "w")

    for epoch in range(args.epoch):  # 循环每个epoch
        # 训练一个epoch
        train_losses, train_acces = train_one_epoch(model, optimizer, train_loader)
        # 在测试集上评估模型
        test_losses, test_acces = evaluation(model, test_loader)

        # 记录损失和准确率
        train_losses_list.append(train_losses)
        train_acces_list.append(train_acces)
        test_losses_list.append(test_losses)
        test_acces_list.append(test_acces)

        # 检查是否得到更好的准确率
        if test_acces > best_acc:
            best_acc = test_acces
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=test_acces)
            name = os.path.join(exp_path, "ckpt", "seed water eem")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            torch.save(state_dict, name)  # 保存模型状态

        # 更新学习率
        scheduler.step()

        # 写入日志文件
        if (epoch + 1) % args.print_freq == 0:
            msg = "epoch:{} model:{} train loss:{:.2f} acc:{:.2f} test loss{:.2f} acc:{:.2f}".format(
                epoch + 1,
                args.model_names,
                train_losses,
                train_acces,
                test_losses,
                test_acces,
            )
            print(msg)  # 打印信息到控制台
            f.write(msg + '\n')  # 写入信息到文件
            f.flush()  # 刷新文件缓冲区

    # 关闭日志文件
    f.close()

    # 使用matplotlib绘制损失与准确率曲线
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epoch + 1), train_losses_list, label='Train Loss')
    plt.plot(range(1, args.epoch + 1), test_losses_list, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 设置x轴的刻度为整数
    plt.xticks(range(1, args.epoch + 1))
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epoch + 1), train_acces_list, label='Train Acc')
    plt.plot(range(1, args.epoch + 1), test_acces_list, label='Test Acc')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # 设置x轴的刻度为整数
    plt.xticks(range(1, args.epoch + 1))

    plt.tight_layout()
    plt.show()

    # 记录最佳准确率和总训练时间
    msg_best = "model:{} best acc:{:.2f}".format(args.model_names, best_acc)
    time_elapsed = "training time: {}".format(time.time() - since)
    print(msg_best)  # 打印信息到控制台
    f.write(msg_best + '\n')  # 写入信息到文件
    f.write(time_elapsed + '\n')  # 写入信息到文件


if __name__ == "__main__":
    base_path = r"D:\HZAU\exp\EEM\CNN\runs"
    tb_path = os.path.join(base_path, args.dataset, args.model_names, args.exp_postfix)
    tb_writer = SummaryWriter(log_dir=tb_path)  # 创建TensorBoard写入器

    # 初始化模型和优化器
    lr = args.lr
    model = model_dict[args.model_names](num_classes=args.classes_num, pretrained=args.pre_trained)
    if torch.cuda.is_available():  # 如果有可用的GPU，则将模型转移到GPU上
        model = model.cuda()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)  # 初始化学习率调度器

    # 开始训练
    train(model, optimizer, train_loader, test_loader, scheduler)