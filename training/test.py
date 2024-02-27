import os
import argparse
from detectors import DETECTOR
import torch
from torch.utils.data import DataLoader
from dataset.datasets_train import *
import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str,
                        default="/ff++/test.csv")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="size of the batches")
    parser.add_argument("--checkpoints", type=str,
                        default="checkpoints/checkpoint.pth")
    parser.add_argument("--inter_attribute", type=str,
                        default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others')
    parser.add_argument("--single_attribute", type=str,
                        default='male-nonmale-asian-white-black-others')
    parser.add_argument("--test_data_name", type=str,
                        default='ff++')
    parser.add_argument("--savepath", type=str,
                        default='/results')
    parser.add_argument("--model_structure", type=str, default='fair_df_detector',
                        help="detector name")

    opt = parser.parse_args()
    print(opt, '!!!!!!!!!!!')

    cuda = True if torch.cuda.is_available() else False
    from transform import fair_df_default_data_transforms as data_transforms


    # prepare the model (detector)
    model_class = DETECTOR['fair_df_detector']
    model = model_class()
    if cuda:
        model.cuda()

    ckpt = torch.load(opt.checkpoints, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt, strict=True)
    print('loading from: ', opt.checkpoints)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    interattributes = opt.inter_attribute.split('-')
    singleattributes = opt.single_attribute.split('-')

    nonmale_dst = []
    male_dst = []
    black_dst = []
    white_dst = []
    others_dst = []
    asian_dst = []

    for eachatt in interattributes:
        print(opt.test_path)
        test_dataset = ImageDataset_Test(
            opt.test_path, eachatt, data_transforms['test'], opt.test_data_name)
        if 'nonmale,' in eachatt:
            nonmale_dst.append(test_dataset)
        else:
            male_dst.append(test_dataset)
        if ',black' in eachatt:
            black_dst.append(test_dataset)
        if ',white' in eachatt:
            white_dst.append(test_dataset)
        if ',others' in eachatt:
            others_dst.append(test_dataset)
        if ',asian' in eachatt:
            asian_dst.append(test_dataset)

        test_dataloader = DataLoader(
            test_dataset, batch_size=opt.batch_size, shuffle=False)

        print("%s" % opt.test_path)
        print('Testing: ', eachatt)
        print('-' * 10)
        print('%d batches int total' % len(test_dataloader))

        corrects = 0.0
        predict = {}
        start_time = time.time()

        pred_list = []
        label_list = []
        face_list = []
        feature_list = []

        for i, data_dict in enumerate(test_dataloader):
            bSTime = time.time()
            model.eval()
            data, label = data_dict['image'], data_dict["label"]
            data_dict['image'], data_dict["label"] = data.to(
                device), label.to(device)

            with torch.no_grad():

                output = model(data_dict, inference=True)
                pred = output['cls_fused'][:, 1]
                pred = pred.cpu().data.numpy().tolist()

                simp_label = label
                pred_list += pred
                label_list += label.cpu().data.numpy().tolist()

            bETime = time.time()
            print('#{} batch finished, eclipse time: {}'.format(i, bETime-bSTime))

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)

        savepath = opt.savepath + '/' + eachatt
        np.save(savepath+'labels.npy', label_list)
        np.save(savepath+'predictions.npy', pred_list)


    print()
    print('-' * 10)
