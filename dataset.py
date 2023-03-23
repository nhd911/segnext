import os
import random
import glob
import sys
import cv2
import tqdm
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from utils import *
from augmentation import RandAugment
from utils import scale_coords, unnormalize
from configs import *


def load_categories():
    s2i = {sbj: idx for idx, sbj in enumerate(categories)}
    i2s = {idx: sbj for idx, sbj in enumerate(categories)}
    return s2i, i2s


c2i, i2c = load_categories()


def load_data(root, mode):
    images = []
    labels = []

    for image_name in tqdm.tqdm(os.listdir(os.path.join(root, mode, 'data')), desc='Preparing'):
        label_name = image_name.split(".")[0] + ".txt"
        img = Image.open(os.path.join(
            root, mode, 'data', image_name)).convert('RGB')
        img = ImageOps.exif_transpose(img)
        images.append(img)

        ann_data = dict()
        with open(os.path.join(root, mode, 'label', label_name), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                field_name, x1, y1, x2, y2 = line.strip().split(' ')
                if field_name not in ann_data.keys():
                    ann_data[field_name] = [[x1, y1, x2, y2]]
                else:
                    ann_data[field_name] += [[x1, y1, x2, y2]]
        labels.append(ann_data)
    print(f"Loaded {len(images)} images {mode}")

    return images, labels


class Resize:
    def __init__(self, nsize):
        self.nsize = nsize

    def _resize_mask(self, mask, new_size):
        foreground = cv2.resize(mask, new_size)
        background = np.zeros((self.nsize[1], self.nsize[0]))
        background[: foreground.shape[0], : foreground.shape[1]] = foreground
        return background

    def __call__(self, image, coors=None, mask=None):
        image_ = image
        factor_x = image.width / self.nsize[0]
        factor_y = image.height / self.nsize[1]
        factor = max(factor_x, factor_y)
        new_size = (min(self.nsize[0], int(image.width / factor)),
                    min(self.nsize[1], int(image.height / factor)))
        new_size = self.nsize
        image = image.resize(size=new_size)
        new_image = Image.new('RGB', self.nsize, color=(0, 0, 0))
        new_image.paste(image, (0, (self.nsize[1] - new_size[1]) // 2))
        if coors is not None:
            coors_new = []
            for coor in coors:
                if len(coor) == 0:
                    coors_new.append([])
                else:
                    boxes = []
                    for box in coor:
                        x1, y1, x2, y2 = int(box[0]), int(
                            box[1]), int(box[2]), int(box[3])
                        box = [x1, y1, x2, y2]
                        box_ = scale_coords(
                            (image_.height, image_.width), box, (new_size[1], new_size[0]))
                        x1, y1, x2, y2 = box_
                        box_ = [x1, y1, x2, y2]
                        boxes.append(box_)
                    coors_new.append(boxes)

            if mask is not None:
                result = multi_apply(self._resize_mask, [m for m in mask], [
                                     new_size] * mask.shape[0])
                mask = np.stack(result, axis=0)
                return new_image, mask, coors_new
            else:
                return new_image, coors_new
        else:
            if mask is not None:
                result = multi_apply(self._resize_mask, [m for m in mask], [
                                     new_size] * mask.shape[0])
                mask = np.stack(result, axis=0)
                return new_image, mask
            else:
                return new_image


class DataTransformer:
    def __init__(self, opt, mode='train'):
        self.width = 500  # chieu rong ban dau
        self.height = 350  # chieu dai ban dau
        self.opt = opt
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        self.randaug = RandAugment(3)
        self.resize = Resize((opt.imgW, opt.imgH))
        self.cate_sort = categories

    def _draw_bounding_mask(self, bboxes):
        mask = Image.new('L', (self.width, self.height), color=0)
        drawer = ImageDraw.Draw(mask)
        for bbox in bboxes:
            bbox = [int(float(i)) for i in bbox]  # convert to integer
            drawer.rectangle(bbox, fill=1)
        return np.array(mask)

    def __call__(self, image, gt=None):
        self.width = image.width
        self.height = image.height
        if gt is not None:
            if len(gt.keys()) != len(self.cate_sort):
                for i in self.cate_sort:
                    if i not in gt.keys():
                        gt[i] = []
            coordinates = []
            for i in self.cate_sort:
                coordinates.append(gt[i])
            gt_masks = multi_apply(self._draw_bounding_mask, coordinates)
            gt_masks = np.stack(gt_masks, axis=0)
            gt_masks = np.array(gt_masks, dtype=np.float)
            image, gt_masks, gt = self.resize(image, coordinates, gt_masks)
            gt_masks = torch.as_tensor(gt_masks).bool().float()
            image = self.transform(image)
            return image, gt_masks[:, ::8, ::8], gt
        else:
            image = self.resize(image)
            image = self.transform(image)
            return image

class DetectorDataset(Dataset):
    def __init__(self, opt, mode='train') -> None:
        super().__init__()
        self.opt = opt
        self.mode = mode

        self.transforms = DataTransformer(self.opt, mode=self.mode)
        self.images, self.labels = load_data(
            root=opt.root_data, mode=self.mode)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index == len(self):
            raise IndexError
        img = self.images[index]
        label = self.labels[index]

        img, gt_masks, gt = self.transforms(img, label)

        return img, gt_masks, gt

def collate_fn(batch):
    image = torch.stack([sample[0] for sample in batch], dim=0)
    gt_masks = torch.stack([sample[1] for sample in batch], dim=0)
    # gt = torch.stack([sample[2] for sample in batch], dim=0)
    return image, gt_masks

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data', required=True,
                        help='path to root dataset')
    parser.add_argument('--imgH', type=int, default=700,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=1000,
                        help='the width of the input image')
    opt = parser.parse_args()

    dataset = DetectorDataset(opt, mode='train')
    image, gt_masks, coor = dataset.__getitem__(15)
    print(image.shape)
    print(gt_masks.shape)
    print(coor)

    image = unnormalize(image.unsqueeze(dim=0), mean=(
        0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = image[0].permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype('uint8')

    gt_masks = gt_masks.cpu().numpy()
    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].imshow((gt_masks == 1).sum(axis=0).reshape(88, 125, 1))
    axs[1].axis('off')
    fig.tight_layout()
    # plt.show()
    # plt.close('all')
    plt.savefig("1.png")
