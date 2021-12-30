import os

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as patches
import xml.etree.ElementTree as ET


def read_annot(file_name, xml_dir):
    """
    Function used to get the bounding boxes and labels from the xml file
    Input:
        file_name: image file name
        xml_dir: directory of xml file
    Return:
        bbox : list of bounding boxes
        labels: list of labels
    """
    bbox = []
    labels = []

    annot_path = os.path.join(xml_dir, file_name[:-3] + 'xml')
    tree = ET.parse(annot_path)
    root = tree.getroot()
    for boxes in root.iter('object'):
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        label = boxes.find('name').text
        bbox.append([xmin, ymin, xmax, ymax])
        if label == 'with_mask':
            label_idx = 2
        else:
            label_idx = 1
        labels.append(label_idx)

    return bbox, labels


# help function for drawing bounding boxes on image
def draw_boxes(img, boxes, labels, thickness=1):
    """
    Function to draw bounding boxes
    Input:
        img: array of img (h, w ,c)
        boxes: list of boxes (int)
        labels: list of labels (int)

    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box, label in zip(boxes, labels):
        box = [int(x) for x in box]
        if label == 2:
            color = (0, 225, 0)  # green
        elif label == 1:
            color = (0, 0, 225)  # red
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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


def single_img_predict(img, model, nm_thrs=0.3, score_thrs=0.8):
    test_img = transforms.ToTensor()(img)
    model.eval()

    with torch.no_grad():
        predictions = model(test_img.unsqueeze(0).to(device))

    test_img = test_img.permute(1, 2, 0).numpy()

    # non-max supression
    keep_boxes = torchvision.ops.nms(predictions[0]['boxes'].cpu(), predictions[0]['scores'].cpu(), nm_thrs)

    # Only display the bounding boxes which higher than the threshold
    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs

    # get the filtered result
    test_boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]

    return test_img, test_boxes, test_labels


def plot_img(img, predict, annotation):
    plt.close()
    fig, ax = plt.subplots(1, 2)
    img = img.cpu().data

    ax[0].imshow(img.permute(1, 2, 0))  # rgb, w, h => w, h, rgb
    ax[1].imshow(img.permute(1, 2, 0))
    ax[0].set_title("ground_truth")
    ax[1].set_title("predict")

    anno_boxs = annotation["boxes"]
    anno_labels = annotation["labels"]
    for i in range(len(anno_boxs)):
        xmin, ymin, xmax, ymax = anno_boxs[i]
        if anno_labels[i] == 2:
            color = 'g'
        elif anno_labels[i] == 1:
            color = 'r'
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax[0].add_patch(rect)
    pre_boxs = predict["boxes"]
    pre_labels = predict["labels"]
    pre_scores = predict["scores"]
    for i in range(len(pre_boxs)):
        if pre_scores[i] < 0.8:
            continue

        xmin, ymin, xmax, ymax = pre_boxs[i]

        if pre_labels[i] == 2:
            color = 'g'
        elif pre_labels[i] == 1:
            color = 'r'
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=color,
                                 facecolor='none')
        ax[1].add_patch(rect)


if __name__ == '__main__':
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

    with torch.no_grad():
        a = 0
        for imgs, annotations in tt_set:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            # Prediction
            preds = model(imgs)

            # test_img, test_boxes, test_labels = single_img_predict(imgs,model)
            # test_output = draw_boxes(test_img, test_boxes, test_labels)

            # Draw the bounding box of ground truth
            # bbox, labels = read_annot(file_list[idx], config['xml_path'])
            # draw bounding boxes on the image
            # gt_output = draw_boxes(test_img, annotations[0]['boxes'], annotations[0]['label'])

            # for i in range(2):
            plot_img(imgs[0], preds[0], annotations[0])
            # Display the result

            # display result
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            # ax1.imshow(test_output)
            # ax1.set_xlabel('Prediction')
            # ax2.imshow(gt_output)
            # ax2.set_xlabel('Ground Truth')

            s = "%d.png" % a
            plt.savefig("../data_set/predict/" + s)
            a = a + 1
