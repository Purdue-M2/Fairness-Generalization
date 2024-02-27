import sys
from detectors import DETECTOR
import torch

from torch.optim import lr_scheduler
import numpy as np
import os.path as osp

from log_utils import Logger
from utils.bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM

import torch.backends.cudnn as cudnn
from dataset.pair_dataset import pairDataset
from scipy.special import softmax
import csv
import argparse

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


parser = argparse.ArgumentParser("Example")
parser.add_argument('--lamda1', type=float, default=0.1,
                    help="alpha_i in bi-level-loss, (0.0~1.0)")
parser.add_argument('--lamda2', type=float, default=0.01,
                    help="alpha in bi-level-loss,(0.0~1.0)")
parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--batchsize', type=int, default=16, help="batch size")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--dataname', type=str, default='ff++',
                    help='ff++, celebdf, dfd, dfdc')
parser.add_argument('--fake_datapath', type=str,
                    default='dataset/ff++')
parser.add_argument('--real_datapath', type=str,
                    default='dataset/ff++')
parser.add_argument("--continue_train", default=False,action='store_true')
parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='fair_df_detector',
                    help="detector name[fair_df_detector]")

args = parser.parse_args()


###### import data transform #######
from transform import fair_df_default_data_transforms as data_transforms

###### load data ######
face_dataset = {x: pairDataset(args.fake_datapath+'fake'+'{}.csv'.format(
    x), args.real_datapath+'real'+'{}.csv'.format(
    x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(
    dataset=face_dataset[x], batch_size=args.batchsize, shuffle=True, num_workers=8, collate_fn=face_dataset[x].collate_fn) for x in ['train', 'val']}
dataset_sizes = {x: len(face_dataset[x]) for x in ['train', 'val']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# prepare the model (detector)
model_class = DETECTOR['fair_df_detector']


def classification_metrics(label, prediction):
    auc = roc_auc_score(label, prediction)
    CM = confusion_matrix(label, prediction >= 0.5)
    acc = accuracy_score(label, prediction >= 0.5)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = FP/(FP+TN)
    TPR = TP/(TP+FN)
    return auc, TPR, FPR, acc

##### calcuate ffpr score calculation during val ################


def cal_ffpr_score(preds, labels, intersec_label):
    efpr_s = 0.0

    logits_idx = (labels < 0.5)
    if np.sum(logits_idx) > 0:
        allg = np.sum(preds[logits_idx] == 1)/np.sum(logits_idx)
    else:
        allg = 0.0
        print("no real data in this batch")

    for j in list(np.unique(intersec_label)):
        groupa = 0.0
        groupb = np.sum(labels[intersec_label == j] < 0.5)

        if groupb != 0:
            groupa = np.sum(preds[(intersec_label == j) & (logits_idx)] == 1)
            group = groupa/groupb
        else:
            group = 0.0

        efpr_s += np.abs(group - allg)

    return efpr_s

##### calcuate feo score during val#############


def cal_feo_score(preds, labels, intersec_label):
    eo_score_r = 0.0
    eo_score_f = 0.0

    logits_idx_r = (labels < 0.5)
    if np.sum(logits_idx_r) > 0:
        allg_r = np.sum(preds[logits_idx_r] == 1)/np.sum(logits_idx_r)
    else:
        allg_r = 0.0
        print("no real data in this batch")

    for j in range(8):
        groupa_r = 0.0
        groupb_r = np.sum(labels[intersec_label == j] < 0.5)

        if groupb_r != 0:
            groupa_r = np.sum(
                preds[(intersec_label == j) & (logits_idx_r)] == 1)
            group_r = groupa_r/groupb_r
        else:
            group_r = 0.0

        eo_score_r += np.abs(group_r - allg_r)

    logits_idx_f = (labels >= 0.5)
    if np.sum(logits_idx_f) > 0:
        allg_f = np.sum(preds[logits_idx_f] == 1)/np.sum(logits_idx_f)
    else:
        allg_f = 0.0
        print("no real data in this batch")

    for j in range(8):
        groupa_f = 0.0
        groupb_f = np.sum(labels[intersec_label == j] >= 0.5)

        if groupb_f != 0:
            groupa_f = np.sum(
                preds[(intersec_label == j) & (logits_idx_f)] == 1)
            group_f = groupa_f/groupb_f
        else:
            group_f = 0.0

        eo_score_f += np.abs(group_f - allg_f)

    return (eo_score_r + eo_score_f)

###### calculate G_auc during val ##############


def auc_gap(preds, labels, intersec_label):
    auc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            auc_section, _, _, _ = classification_metrics(
                labels_section, pred_section)
            auc_all_sec.append(auc_section)
        except:
            continue
    return max(auc_all_sec)-min(auc_all_sec)


def cal_foae_score(preds, labels, intersec_label):
    acc_all_sec = []

    for j in list(np.unique(intersec_label)):
        pred_section = preds[intersec_label == j]
        labels_section = labels[intersec_label == j]
        try:
            _, _, _, acc_section = classification_metrics(
                labels_section, pred_section)
            acc_all_sec.append(acc_section)
        except:
            continue
    return max(acc_all_sec)-min(acc_all_sec)
# train and evaluation


def train(model,  optimizer, scheduler, num_epochs, start_epoch):

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        for idx, data_dict in enumerate(dataloaders[phase]):

            imgs, labels, intersec_labels = data_dict['image'], data_dict[
                'label'], data_dict['intersec_label']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                device), labels.to(device), intersec_labels.to(device),

           
            # SAM
            # first forward-backward step
            enable_running_stats(model)
            preds = model(data_dict)

            # # first forward-backward pass
            losses = model.get_losses(data_dict, preds)
            losses = losses['overall']
            losses.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            # make sure to do a full forward pass
            preds = model(data_dict)
            losses = model.get_losses(data_dict, preds)
            losses = losses['overall']
        
            losses.backward()
            optimizer.second_step(zero_grad=True)

            if idx % 50 == 0:
                # compute training metric for each batch data
                batch_metrics = model.get_train_metrics(data_dict, preds)
                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += losses.item() * imgs.size(0)

        epoch_loss = total_loss / dataset_sizes[phase]
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
            scheduler.step()

        # evaluation
        if (epoch+1) % 1 == 0:
  
            savepath = './checkpoints/'+args.model+'/'+args.dataname+'_'+'/lamda1_' + \
                    str(args.lamda1)+'_lamda2_' + \
                    str(args.lamda2)+'_lr'+str(args.lr)


            temp_model = savepath+"/"+args.model+str(epoch)+'.pth'
            torch.save(model.state_dict(), temp_model)

            print()
            print('-' * 10)

            phase = 'val'
            model.eval()
            running_corrects = 0
            total = 0

            pred_label_list = []
            pred_probs_list = []
            label_list = []
            intersec_label_list = []

            for idx, data_dict in enumerate(dataloaders[phase]):
                imgs, labels, intersec_labels = data_dict['image'], data_dict['label'], data_dict['intersec_label']
                # do not consider the specific label when testing
                # fix the label to 0 and 1 only
                labels = torch.where(data_dict['label'] != 0, 1, 0)
                if 'label_spe' in data_dict:
                    data_dict.pop('label_spe')  # remove the specific label

                data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                    device), labels.to(device), intersec_labels.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(data_dict, inference=True)
                    _, preds_label = torch.max(preds['cls_fused'], 1)
                    pred_probs = torch.softmax(
                        preds['cls_fused'], dim=1)[:, 1]
                    total += data_dict['label'].size(0)
                    running_corrects += (preds_label ==
                                         data_dict['label']).sum().item()

                    preds_label = preds_label.cpu().data.numpy().tolist()
                    pred_probs = pred_probs.cpu().data.numpy().tolist()
                
                pred_label_list += preds_label
                pred_probs_list += pred_probs
                label_list += labels.cpu().data.numpy().tolist()
                intersec_label_list += intersec_labels.cpu().data.numpy().tolist()
                if idx % 50 == 0:
                    batch_metrics = model.get_test_metrics()
                    print('#{} batch_metric{{"acc": {}, "auc": {}, "eer": {}, "ap": {}}}'.format(idx,
                                                                                                 batch_metrics['acc'],
                                                                                                 batch_metrics['auc'],
                                                                                                 batch_metrics['eer'],
                                                                                                 batch_metrics['ap']))

            pred_label_list = np.array(pred_label_list)

            pred_probs_list = np.array(pred_probs_list)
            label_list = np.array(label_list)
            intersec_label_list = np.array(intersec_label_list)

            epoch_acc = running_corrects / total

            feo_score = cal_feo_score(
                pred_label_list, label_list, intersec_label_list)
            foae_score = cal_foae_score(
                pred_probs_list, label_list, intersec_label_list)
            auc_maxgap = auc_gap(
                pred_probs_list, label_list, intersec_label_list)
            auc, TPR, FPR, _ = classification_metrics(
                label_list, pred_probs_list)

            print('Epoch {} Acc: {:.4f} foae score: {:.4f} feo score: {} auc: {}, tpr: {}, fpr: {}'.format(
                epoch, epoch_acc, foae_score, feo_score, auc, TPR, FPR))
            with open(savepath+"/val_metrics.csv", 'a', newline='') as csvfile:
                columnname = ['epoch', 'epoch_acc', 'foae_score', 'feo_score',
                              'auc_gap_inter', 'AUC all', 'TPR all', 'FPR all']
                writer = csv.DictWriter(csvfile, fieldnames=columnname)
                writer.writerow({'epoch': str(epoch), 'epoch_acc': str(epoch_acc), 'foae_score': str(foae_score), 'feo_score': str(
                    feo_score), 'auc_gap_inter': str(auc_maxgap), 'AUC all': str(auc), 'TPR all': str(TPR), 'FPR all': str(FPR)})

            print()
            print('-' * 10)

    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()


    sys.stdout = Logger(osp.join('./checkpoints/'+args.model+'/'+args.dataname+'_'+'/lamda1_'+str(
            args.lamda1)+'_lamda2_'+str(args.lamda2)+'_lr'+str(args.lr)+'/log_training.txt'))


    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)


    start_epoch = 0

    if args.continue_train and args.checkpoints != '':
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict)
        print('continue train from: ', args.checkpoints)
        start_epoch = 49

    # optimize
    params_to_update = model.parameters()

    base_optimizer = torch.optim.SGD
    optimizer = SAM(params_to_update, base_optimizer,
                    lr=args.lr, momentum=0.9, weight_decay=5e-03)
    print(params_to_update, optimizer)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=60, gamma=0.9)


    model, epoch = train(model, optimizer,
                         exp_lr_scheduler, num_epochs=100, start_epoch=start_epoch)

    if epoch == 99:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
