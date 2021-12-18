import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as patches

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




def plot_img(img, predict, annotation):
    fig, ax = plt.subplots(1, 2)
    img = img.cpu().data

    ax[0].imshow(img.permute(1, 2, 0))  # rgb, w, h => w, h, rgb
    ax[1].imshow(img.permute(1, 2, 0))
    ax[0].set_title("real")
    ax[1].set_title("predict")

    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax[0].add_patch(rect)

    for box in predict["boxes"]:
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
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
    model = Faster_RCNN()
    model.load_state_dict(torch.load(PATH))
    model.eval()  # 当用于inference时不要忘记添加
    # train_set = np.load("checkpoint/train_set.npy")
    test_set = np.load("checkpoint/test_set.npy")

    # tr_set = prep_dataloader(train_set, config['xml_path'], 'train', config['batch_size'], config['n_jobs'])
    tt_set = prep_dataloader(test_set, config['xml_path'], 'test', config['batch_size'], config['n_jobs'])
    with torch.no_grad():
        for imgs, annotations in tt_set:
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            preds = model(imgs)

            for i in range(len(imgs)):
                plot_img(imgs[i], preds[i], annotations[i])
                # plot_img(imgs[i], annotations[i])
                s = i + ".png"
                plt.savefig("../data_set/predict/" + s)
            break