#coding=utf-8
import itertools
from collections import defaultdict

import torch
import torchvision
from six.moves import zip
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

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
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)
    return model


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    with torch.no_grad():
        for imgs, annotations in dataloader:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            # for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in (enumerate(dataloader)):
            #““” 将每个图片预测出来的结果放在图片的一个列表中“””
            # sizes = [sizes[0][0], sizes[1][0]]
            # pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
            # Prediction
            predict = model(imgs)

            #将真实的目标框、类别、difficults存入list
            gt_bboxes_ = annotations[0]["boxes"]
            gt_labels_ = annotations[0]["labels"]
            gt_bboxes += gt_bboxes_
            gt_labels += gt_labels_
            # gt_difficults += list(gt_difficults_.numpy())
            #将预测的目标框、类别、分数存入list
            pred_bboxes_ = predict[0]["boxes"]
            pred_labels_ = predict[0]["labels"]
            pred_scores_ = predict[0]["scores"]

            pred_bboxes += pred_bboxes_
            pred_labels += pred_labels_
            pred_scores += pred_scores_
            # if ii == test_num: break
        #返回dictz字典，两个key值：AP、mAP
        result = eval_detection_voc(
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels,use_07_metric=True)
    return result


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}

def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None)
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            (
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):
        """先对每个图片进行循环"""

        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            """循环每个类别"""
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:  # 如果没有预测这个类别的 直接循环下一个类别了
                continue
            if len(gt_bbox_l) == 0:  # 如果gt没有这个类别，还预测了这个类别，match填入0，没匹配上
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            iou = IOU(pred_bbox_l, gt_bbox_l)  # 对于每个图片，类别正确 才开始计算iou，iou>阈值 才说明正确
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:  # 这里是每个gt只匹配一次，下面图片说明
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
            # pred_bboxes, pred_labels, pred_scores,
            # gt_bboxes, gt_labels):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    计算AP值，若use_07_metric=true,则用11个点采样的方法，将rec从0-1分成11个点，这些点prec值求平均近似表示AP
    若use_07_metric=false,则采用更为精确的逐点积分方法
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def IOU(bbox, gt):
    """
    left-top point and right-bottom point
    :param bbox: bounding box
    :param gt: ground truth
    :return: iou = intersection / union
    """
    min_x = max(bbox[0], gt[0])
    min_y = max(bbox[1], gt[1])
    max_x = min(bbox[2], gt[2])
    max_y = min(bbox[3], gt[3])
    width = max(0, max_x - min_x + 1)
    height = max(0, max_y - min_y + 1)

    inters = width * height
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)

    union = bbox_area + gt_area - inters
    overlaps = inters / union
    return overlaps

if __name__ =='__main__':
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

    result = eval(tt_set, model, test_num=10000)
    print(result)