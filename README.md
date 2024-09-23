# 基于CNN和3D-EEM分析玉米种子水的差异
##  3D-EEM
### 原始数据获取
此次实验数据是使用华中农业大学资环学院F-7100 荧光分度计（日立）所得，其中仪器参数
EX：200-400；  EM：270_500；slit：5；  scan speed：2400nm/min；  PMT voltage：600V。
### 原始数据处理
原始数据以及处理的相关代码在seed EEM文件夹
使用EEM-MAP.py进行背景值去除以及散色消除，并且绘制三维荧光光谱热图。
## 通过CNN训练EEM模型 
### 软件配置
Python                    3.11.9
torch                     2.4.1+cpu
torchaudio                2.4.1+cpu
torchvision               0.19.1+cpu
pandas                    2.2.2
这个网络模型基于ResNet18架构，未使用预训练权重，
针对含有30个类别的数据集进行训练，配置了SGD优化器（学习率为0.01，动量为0.9，权重衰减为1e-4），
并在20个训练周期内使用批大小为64进行迭代，训练过程使用cpu执行，每完成一个周期打印一次训练信息，
为确保实验的可重复性，设置了随机种子33。
### 软件运行
配置好环境后在IDE运行train.py ,
或者在conda命令端口输入 python train.py --model resnet18  --batch_size 64 --lr 0.001 --epoch 100 --classes_num 30
运行该程序。
