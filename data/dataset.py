import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms as T
import pandas as pd
from config import opt
from tqdm import tqdm


# 使用`Dataset`提供数据集的封装，再使用`Dataloader`实现数据并行加载。
class DogBreedData(Dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        super(DogBreedData, self).__init__()
        """
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]# imgs是图片地址的list
        imgs = sorted(imgs, key=lambda x: (x.split('.')[-2])) # 将图片按id进行排序，因为对应的labels是按图片id排序的

        # 对labels进行处理，转换成one-hot向量
        labels = pd.read_csv('data/labels.csv')
        labels['target'] = 1
        labels = labels.pivot('id', 'breed', 'target')
        labels = labels.reset_index().fillna(0)
        self.labels = labels.drop('id',axis=1, inplace=False)
        self.labels = self.labels.values.argmax(axis=1)   # (N,) 元素是[0,C) C类

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.05*imgs_num)]   # train data
            self.labels = self.labels[:int(0.05*imgs_num)]
        else:
            self.imgs = imgs[int(0.95*imgs_num):]    # val data
            self.labels = self.labels[int(0.95 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])


    def __getitem__(self, index):
        img_path = self.imgs[index]
        if not self.test:
            labels = self.labels[index]
        else:
            labels = self.imgs[index].split('.')[-2]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, labels

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    print(__file__)
    labels = pd.read_csv('labels.csv')
    train_data = DogBreedData(root=opt.train_data_root,transforms=True, train=False)  #(data, labels)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)
    for ii, (data, label) in tqdm(enumerate(train_dataloader)):
        print(data.shape, label.shape) # torch.Size([128, 3, 224, 224]) torch.Size([128,])



