# 写文章
# 点击打开以后的主页
# 计算图像数据集的均值与方差
# freethinker
# freethinker
# 专注于计算机视觉，图像复原与重建
# ​关注
# 5
# 人赞同了该文章
# Pytorch进行预处理时，通常使用torchvision.transforms.Normalize(mean, std)
# 方法进行数据归一化，其中参数mean和std分别表示图像集每个通道的均值和方差序列。
#
# 在训练Imagenet数据集时通常设置：mean = (0.485, 0.456, 0.406)，std = (0.229, 0.224, 0.225)。而对于特定的数据集，选择这个值结果可能并不理想。接下来给出计算特定数据集的均值和方差的方法。
import torch
from torchvision.datasets.folder import ImageFolder


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_dataset = ImageFolder(root='./img', transform=None)
    print(getStat(train_dataset))
