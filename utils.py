import math, cv2
import torch
from functools import partial
import numpy as np


def unnormalize(image, mean, std):
    image = ((image * torch.as_tensor(std).reshape(1, image.size(1), 1, 1).to(image.device)) + torch.as_tensor(mean).reshape(1, image.size(1), 1, 1).to(image.device))
    return image

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    results = map(pfunc, *args)
    return list(results)

def rotate_bbox(bboxes, radians, anchors):
    rotate_mat = torch.stack(
        [torch.cos(radians), -torch.sin(radians), torch.sin(radians), torch.cos(radians)],
    dim=-1) # (N, 4)
    rotate_mat = rotate_mat.reshape(rotate_mat.size(0), 2, 2) # (N, 2, 2)
    anchors = anchors.unsqueeze(dim=1) # (N, 1, 2)
    rotated_bboxes = anchors + torch.matmul(rotate_mat, (bboxes - anchors).transpose(1, 2)).transpose(1, 2)
    return rotated_bboxes

def scale_output(bboxes, old_size, new_size, norm_size):
    bboxes -= np.array(((norm_size[0] - new_size[0]) // 2, (norm_size[1] - new_size[1]) // 2)).reshape((1, 1, -1))
    bboxes = bboxes.astype(np.float32)
    bboxes /= (np.array(new_size).astype(np.float32) / np.array(old_size).astype(np.float32)).reshape((1, 1, -1))
    bboxes = bboxes.astype(np.int32)
    return bboxes

def normalize(x, dim=None):
    if not dim:
        return (x - x.min()) / (x.max() - x.min())
    else:
        return (x - x.min(dim=dim, keepdims=True)[0]) / (x.max(dim=dim, keepdims=True)[0] - x.min(dim=dim, keepdims=True)[0])

def crop_by_transform(image, bboxes_1, bboxes_2, dst=(1000, 700)):
    bboxes_1, bboxes_2 = bboxes_1.astype(np.float32), bboxes_2.astype(np.float32)
    matrix = cv2.getPerspectiveTransform(bboxes_1, bboxes_2)
    cropped = cv2.warpPerspective(image, matrix, dst)
    return cropped

def post_process(bmap):
    contours, _ = cv2.findContours(bmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        output.append((x * 8, y * 8, (x+w) * 8, (y+h) * 8))
    return output

def scale_coords(img1_shape, coords, img0_shape):
    x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
    factor_x = img0_shape[1] / img1_shape[1]
    factor_y = img0_shape[0] / img1_shape[0]

    return [int(x1 * factor_x), int(y1 * factor_y), int(x2 * factor_x), int(y2 * factor_y)]