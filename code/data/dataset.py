# coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import random


class FaceAttack(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs_tmp_real = [os.path.join(root, 'real', tmp) for tmp in os.listdir(os.path.join(root, 'real'))]
        imgs_tmp_attack_fixed = [os.path.join(root, 'attack', 'fixed', tmp) for tmp in os.listdir(os.path.join(root, 'attack', 'fixed'))]
        imgs_tmp_attack_hand = [os.path.join(root, 'attack', 'hand', tmp) for tmp in os.listdir(os.path.join(root, 'attack', 'hand'))]
        imgs_tmp = imgs_tmp_real + imgs_tmp_attack_fixed + imgs_tmp_attack_hand
        imgs = []
        for tmp in imgs_tmp:
            imgs.extend([os.path.join(tmp, img) for img in os.listdir(tmp)])

        imgs_num = len(imgs)
        
        random.shuffle(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

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
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            label = 1 if 'real' in img_path.split('/') else 0
        else:
            label = 1 if 'real' in img_path.split('/') else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
