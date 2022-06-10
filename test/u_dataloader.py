import os
import numpy as np
# import random
import matplotlib.pyplot as plt
# import collections
import torch
import torchvision
# import cv2
from PIL import Image


# import torchvision.transforms as transforms
# 自定义dataset

class U_DataSet(torch.utils.data.Dataset):
    def __init__(self, root, ignore_label=255):
        super(U_DataSet, self).__init__()
        self.root = root
        # self.list_path = list_path
        # self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = [n[:-4] for n in os.listdir(root + "/img")]
        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.root, "img/%s.png" % name)
            label_file = os.path.join(self.root, "labelcol/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        '''load the datas'''
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"]).convert('L')
        size_origin = image.size  # W * H

        I = np.asarray(image, np.float32)

        I = I.transpose((2, 0, 1))  # transpose the  H*W*C to C*H*W
        L = np.asarray(np.array(label), np.int64)
        # print(I.shape,L.shape)
        return I.copy(), L.copy(), np.array(size_origin), name


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    # 更加dataset初始化dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    # 便利dataloader，计算训练集的矩阵和标准差
    for X, _, _, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':

    Batch_size = 1
    # 初始化数据集
    dst = U_DataSet('./Train/')
    # 开始计算
    print(getStat(dst))
    # 下边代码不是计算均值和标准差的代码
    # plt.ion()
    # trainloader = torch.utils.data.DataLoader(dst, batch_size=Batch_size)
    # for i, data in enumerate(trainloader):
    #     imgs, labels, _, _ = data
    #     if i % 1 == 0:
    #         img = torchvision.utils.make_grid(imgs).numpy()
    #         img = img.astype(np.uint8)  # change the dtype from float32 to uint8,
    #         # because the plt.imshow() need the uint8
    #         img = np.transpose(img, (1, 2, 0))  # transpose the C*H*W to H*W*C
    #         # img = img[:, :, ::-1]
    #         plt.imshow(img)
    #         plt.show()
    #         plt.pause(0.5)
    #         for i in range(labels.shape[0]):
    #             plt.imshow(labels[i], cmap='gray')
    #             plt.show()
    #             plt.pause(0.5)

