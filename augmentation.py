import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
from PIL import Image
import cv2
from utils import multi_apply


class RandAugment:
    def __init__(self, n):
        self._n = n

        self._augment_list = [
            self.HorizontalMotionBlur,
            self.VerticalMotionBlur,
            self.Rotation,
            self.AutoContrast,
            self.Equalize,
            self.Invert,
            self.Posterize,
            self.SolarizeAdd,
            self.Color,
            self.Contrast,
            self.Brightness,
            self.ScaleRand,
            self.Perspective
        ]

    def HorizontalMotionBlur(self, img, mask):
        ksize = np.random.randint(1, 3)
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
        kernel /= ksize
        img = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(img), mask

    def VerticalMotionBlur(self, img, mask):
        ksize = np.random.randint(1, 3)
        kernel = np.zeros((ksize, ksize))
        kernel[:, int((ksize - 1) / 2)] = np.ones(ksize)
        kernel /= ksize
        img = cv2.filter2D(np.array(img), -1, kernel)
        return Image.fromarray(img), mask

    def ScaleRand(self, img, mask):
        def _mask_resize(mask, v):
            return cv2.resize(mask, (int(mask.shape[1] * v), mask.shape[0]))

        v = 1 - np.random.choice(np.arange(0, 0.2, 0.03))
        img = img.resize((int(img.width * v), img.height))

        result = multi_apply(_mask_resize, [m for m in mask], [v] * mask.shape[0])
        mask = np.stack(result, axis=0)
        return img, mask

    def Rotation(self, img, mask):
        def _mask_rotate(mask, angle):
            mask = Image.fromarray(mask).rotate(angle, expand=1)
            return np.array(mask)

        angle = np.random.randint(-10, 10)
        img = img.rotate(angle, expand=1)

        result = multi_apply(_mask_rotate, [m for m in mask], [angle] * mask.shape[0])
        mask = np.stack(result, axis=0)
        return img, mask

    def Perspective(self, img, mask):
        def find_coeffs(source_coords, target_coords):
            matrix = []
            for s, t in zip(source_coords, target_coords):
                matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
                matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
            A = np.matrix(matrix, dtype=np.float)
            B = np.array(source_coords).reshape(8)
            res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
            return np.array(res).reshape(8)

        def _mask_perspective_transform(mask, coeffs):
            mask = Image.fromarray(mask)
            mask = mask.transform(mask.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
            return np.array(mask)

        w = img.width
        h = img.height
        coeffs = find_coeffs(
            [(0, 0), (w, 0), (w, h), (0, h)],
            [
                (0 + np.random.choice(np.arange(-50, 50)), 0 + np.random.choice(np.arange(-50, 50))),
                (w + np.random.choice(np.arange(-50, 50)), 0 + np.random.choice(np.arange(-50, 50))),
                (w + np.random.choice(np.arange(-50, 50)), h + np.random.choice(np.arange(-50, 50))),
                (0 + np.random.choice(np.arange(-50, 50)), h + np.random.choice(np.arange(-50, 50)))
            ])

        img = img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)

        result = multi_apply(_mask_perspective_transform, [m for m in mask], [coeffs] * mask.shape[0])
        mask = np.stack(result, axis=0)
        return img, mask

    def AutoContrast(self, img, mask):
        return PIL.ImageOps.autocontrast(img), mask

    def Invert(self, img, mask):
        return PIL.ImageOps.invert(img), mask

    def Equalize(self, img, mask):
        return PIL.ImageOps.equalize(img), mask

    def Solarize(self, img, mask):
        v = np.random.randint(0, 255)
        return PIL.ImageOps.solarize(img, v), mask

    def SolarizeAdd(self, img, mask=None, addition=0, threshold=128):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + addition
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold), mask

    def Posterize(self, img, mask):
        v = np.random.randint(2, 6)
        return PIL.ImageOps.posterize(img, v), mask

    def Contrast(self, img, mask):
        return PIL.ImageEnhance.Contrast(img).enhance(np.random.choice(np.arange(0.1, 1.2, 0.1))), mask

    def Color(self, img, mask):
        return PIL.ImageEnhance.Color(img).enhance(np.random.choice(np.arange(0.1, 1.2, 0.1))), mask

    def Brightness(self, img, mask):
        return PIL.ImageEnhance.Brightness(img).enhance(np.random.choice(np.arange(0.1, 1.2, 0.1))), mask

    def __call__(self, img, mask=None):
        rand = random.random()
        if rand > 0.25:
            ops = np.random.choice(self._augment_list, self._n)
            for op in ops:
                img, mask = op(img, mask)
        return img, mask
