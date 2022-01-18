import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

from typing import List

sys.path.append("utils")
from utils.mean_avg_precision import mean_average_precision


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def collate_fn(batch):
    return tuple(zip(*batch))


def prep_dataloader(mask_dataset, xml_path, mode, batch_size, n_jobs):
    mask_loader = DataLoader(mask_dataset,
                             batch_size=batch_size,
                             shuffle=(mode == 'train'),
                             num_workers=n_jobs,
                             collate_fn=collate_fn)
    return mask_loader


def Faster_RCNN(device):
    num_classes = 3  # background, without_mask, with_mask

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    return model


def parse(predict, annotation):


    anno_boxs = annotation["boxes"]
    anno_labels = annotation["labels"]
    for i in range(len(anno_boxs)):
        xmin, ymin, xmax, ymax = anno_boxs[i]

    pre_boxs = predict["boxes"]
    pre_labels = predict["labels"]
    pre_scores = predict["scores"]
    for i in range(len(pre_boxs)):
        if pre_scores[i] < 0.8:
            continue
        xmin, ymin, xmax, ymax = pre_boxs[i]

def get_map(set):
    with torch.no_grad():
        a = 0
        for imgs, annotations in set:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            preds = model(imgs)
            annotation=annotations[0]
            predict=preds[0]

            anno_boxs = annotation["boxes"]
            anno_labels = annotation["labels"]
            for i in range(len(anno_boxs)):
                xmin, ymin, xmax, ymax = anno_boxs[i]
                l_true =[a, anno_labels[i], 1,xmin, ymin, xmax, ymax]
                true_boxes.append(l_true)

            pre_boxs = predict["boxes"]
            pre_labels = predict["labels"]
            pre_scores = predict["scores"]
            for i in range(len(pre_boxs)):
                xmin, ymin, xmax, ymax = pre_boxs[i]
                l_pre = [a, pre_labels[i], pre_scores[i], xmin, ymin, xmax, ymax]
                pred_bboxes.append(l_pre)
            a + 1
    precisions,recalls,ap = mean_average_precision(pred_bboxes, true_boxes, iou_threshold=0.5,box_format="corners", num_classes=3)

if __name__ == "__main__":
    print("Running Mean Average Precisions Tests:")
    config = {
        'num_epochs': 5,  # maximum number of epochs
        'batch_size': 1,  # mini-batch size for dataloader
        'n_jobs': 2,
        'optimizer': 'SGD',  # optimization algorithm (optimizer in torch.optim)
        # 'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,  # learning rate of SGD
        'momentum': 0.9,  # momentum for SGD
        'weight_decay': 0.0005,
        # },
        'early_stop': 200,  # early stopping epochs (the number epochs since your model's last improvement)
        'dir_path': '../data_set/face_mask_detection/IMAGES',
        'xml_path': '../data_set/face_mask_detection/ANNOTATIONS',
        'save_path': 'models/model.pth'  # your model will be saved here
    }

    PATH = "checkpoint/model.pth"
    device = get_device()
    model = Faster_RCNN(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()  # 当用于inference时不要忘记添加
    # train_set = np.load("checkpoint/train_set.npy")
    test_set = np.load("checkpoint/test_set.npy")

    # tr_set = prep_dataloader(train_set, config['xml_path'], 'train', config['batch_size'], config['n_jobs'])
    tt_set = prep_dataloader(test_set, config['xml_path'], 'test', config['batch_size'], config['n_jobs'])
    # unittest.main()
    pred_bboxes=[]
    true_boxes=[]
    with torch.no_grad():
        a = 0
        for imgs, annotations in tt_set:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            preds = model(imgs)
            annotation=annotations[0]
            predict=preds[0]

            anno_boxs = annotation["boxes"]
            anno_labels = annotation["labels"]
            for i in range(len(anno_boxs)):
                xmin, ymin, xmax, ymax = anno_boxs[i]
                l_true =[a, anno_labels[i], 1,xmin, ymin, xmax, ymax]
                true_boxes.append(l_true)

            pre_boxs = predict["boxes"]
            pre_labels = predict["labels"]
            pre_scores = predict["scores"]
            for i in range(len(pre_boxs)):
                xmin, ymin, xmax, ymax = pre_boxs[i]
                l_pre = [a, pre_labels[i], pre_scores[i], xmin, ymin, xmax, ymax]
                pred_bboxes.append(l_pre)
            a + 1
    precisions,recalls,ap = mean_average_precision(pred_bboxes, true_boxes, iou_threshold=0.5,box_format="corners", num_classes=3)
    plt.plot(recalls,precisions,'b',label='ap=%f'%ap)
    plt.title('precision-recall curve')
    plt.legend(loc="lower left")
    plt.xlim(-0.1, 1.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig("output/ap.png")
    # print(precisions)
    # print(recalls)
    # print(ap)
