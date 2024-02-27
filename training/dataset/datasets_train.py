import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class ImageDataset_Train(Dataset):
    '''
    Data format in .csv file each line:
    /path/to/image.jpg,label,spe_label,male,nonmale,asian,black,white,others,intersectional_index
    '''

    def __init__(self, csv_file, owntransforms, state, name):
        super(ImageDataset_Train, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms
        self.name = name

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path_label.iloc[idx, 0]

        if img_path != 'img_path':
            img = Image.open(img_path)
            img = self.transform(img)
            label = np.array(self.img_path_label.iloc[idx, 1])

            if self.name == 'dfd':
                intersec_label = np.array(self.img_path_label.iloc[idx, 5])
            if self.name == 'ff++':
                intersec_label = np.array(self.img_path_label.iloc[idx, 6])
            if self.name == 'dfdc':
                intersec_label = np.array(self.img_path_label.iloc[idx, 6])
            if self.name == 'celebdf':
                intersec_label = np.array(self.img_path_label.iloc[idx, 5])

        return {'image': img, 'label': label, 'intersec_label': intersec_label}


class ImageDataset_Test(Dataset):
    # def __init__(self, csv_file, img_size, filter_size, test_set):
    def __init__(self, csv_file, attribute, owntransforms, test_set):
        self.transform = owntransforms
        self.img = []
        self.label = []

        att_list = attribute.split(',')

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            line_count = 0
            for row in rows:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    if test_set == 'ff++':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])

                            intersec_label = int(row[6])
                            if len(att_list) == 2:
                                if attribute == 'male,asian':
                                    if intersec_label == 0:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,white':
                                    if intersec_label == 1:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,black':
                                    if intersec_label == 2:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,others':
                                    if intersec_label == 3:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,asian':
                                    if intersec_label == 4:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,white':
                                    if intersec_label == 5:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,black':
                                    if intersec_label == 6:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,others':
                                    if intersec_label == 7:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)

                    if test_set == 'dfd':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            intersec_label = int(row[5])

                            if len(att_list) == 2:
                                if attribute == 'male,white':
                                    if intersec_label == 0:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,black':
                                    if intersec_label == 1:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,others':
                                    if intersec_label == 2:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,white':
                                    if intersec_label == 3:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,black':
                                    if intersec_label == 4:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,others':
                                    if intersec_label == 5:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)

                    if test_set == 'celebdf':
                        img_path = row[0]
                        if img_path != 'img_path':
                            mylabel = int(row[1])
                            intersec_label = int(row[5])
                            if len(att_list) == 2:
                                if attribute == 'male,white':
                                    if intersec_label == 0:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,black':
                                    if intersec_label == 1:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,others':
                                    if intersec_label == 2:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,white':
                                    if intersec_label == 3:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,black':
                                    if intersec_label == 4:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,others':
                                    if intersec_label == 5:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)

                    if test_set == 'dfdc':
                        img_path = row[0]
                        if img_path != 'img_path':
                            intersec_label = int(row[6])
                            mylabel = int(row[1])
                            if len(att_list) == 2:
                                if attribute == 'male,asian':
                                    if intersec_label == 0:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,white':
                                    if intersec_label == 1:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,black':
                                    if intersec_label == 2:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'male,others':
                                    if intersec_label == 3:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,asian':
                                    if intersec_label == 4:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,white':
                                    if intersec_label == 5:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,black':
                                    if intersec_label == 6:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)
                                if attribute == 'nonmale,others':
                                    if intersec_label == 7:
                                        self.img.append(img_path)
                                        self.label.append(mylabel)

        print(attribute, len(self.img), len(self.label))

    def __getitem__(self, index):

        path = self.img[index % len(self.img)]

        img = Image.open(path)
        label = self.label[index % len(self.label)]
        img = self.transform(img)
        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.img)
