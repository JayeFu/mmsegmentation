import os
from typing import Union, List

import mmcv
import numpy as np
from numpy import random
import albumentations as A

from ..builder import PIPELINES


@PIPELINES.register_module()
class ForegroundAugmentor:
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.4):
        self.transform = A.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            always_apply=True, p=1.0  # same effect
        )

    def __call__(self, results):
        img = results['img']
        img_fore_auged = self.transform(image=img)['image']
        results['img'] = img_fore_auged
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness={self.transform.brightness}'
                     f'contrast={self.transform.contrast}'
                     f'saturation={self.transform.saturation}'
                     f'hue={self.transform.hue}')
        return repr_str


@PIPELINES.register_module()
class BackgroundAugmentor:
    def __init__(self, back_path: Union[List[str], str],
                 random_prob=0.4, color_prob=0.4, bg_percent=0.5, noise_level=0.2, confuse_perturb=0.2,
                 num_holes=8, max_size=10, min_size=1,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'
                 ):
        self.original_th = 1 - random_prob - color_prob
        self.random_th = self.original_th + random_prob
        self.bg_percent = bg_percent
        self.noise_level = noise_level
        self.confuse_perturb = confuse_perturb

        self.num_holes = num_holes
        self.cutout = A.CoarseDropout(
            max_holes=1,  # apply this multiple times for multicolor dropout
            max_height=max_size,
            max_width=max_size,
            min_height=min_size,
            min_width=min_size,
            fill_value=(0, 0, 0),
            always_apply=True, p=1.0  # same effect
        )

        back_path = [back_path] if isinstance(back_path, str) else back_path
        self.back_info = self._load_img_info(img_path=back_path)
        assert len(back_path) > 0, f"[EXTERNAL] Found no external background images in {back_path}"
        print("[EXTERNAL] Found {0} external background images in {1}".format(len(self.back_info), back_path))

        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        mask_fore = (results['gt_semantic_seg'] == 0)  # 0 -> gate, 1 -> background
        fore_img = results['img'].copy()
        fore_img[~mask_fore] = 0  # background set to 0

        # --------------- as if normalized in range 0.0 ~ 1.0
        noise = np.random.rand(*results['img'].shape)  # (H, W, 3)
        confuse_color = np.sum((fore_img.astype(float) / 255.0).reshape((-1, 3)), axis=0) / np.sum(mask_fore)  # (3, )
        confuse_color = confuse_color.reshape((1, 1, 3))  # (1, 1, 3)
        random_color = np.random.rand(1, 1, 3)  # (1, 1, 3)

        # Do not add distractor, we do not have mirroring floors

        randomness = np.random.uniform(0, 1)
        random_background = noise * self.noise_level
        if randomness < self.original_th:  # add original image
            random_background += results['img'] / 255.0 * self.bg_percent
        elif randomness < self.random_th:  # add random color
            random_background += random_color * self.bg_percent
        else:  # add confuse color
            random_background += \
                (confuse_color + np.random.rand(*confuse_color.shape) * self.confuse_perturb) * self.bg_percent

        # add random boxes, i.e., multi color cutout
        for _ in range(self.num_holes):
            self.cutout.fill_value = np.random.rand(1, 1, 3)
            random_background = self.cutout(image=random_background)['image']

        # augment with external image
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        external_img_idx = np.random.randint(len(self.back_info))
        external_img_bytes = self.file_client.get(self.back_info[external_img_idx])
        external_img = mmcv.imfrombytes(
            external_img_bytes, flag=self.color_type, backend=self.imdecode_backend
        )
        external_img = mmcv.imresize(external_img, size=random_background.shape[0: 2][::-1])  # resize, otherwise need to pad
        external_img = external_img.astype(float) / 255.0
        alpha = np.random.rand(1)
        random_background = (1 - alpha) * random_background + alpha * external_img
        # ---------------

        # normalize back to 0 ~ 255
        random_background = np.clip(random_background, 0.0, 1.0)
        random_background *= 255.0
        random_background = np.round(random_background).astype(np.uint8)

        aug_img = fore_img
        aug_img[~mask_fore] = random_background[~mask_fore]

        results['img'] = aug_img
        return results

    @staticmethod
    def _load_img_info(img_path: List[str], img_extensions=('.jpg', '.png', '.jpeg')):
        f_info = []
        for dir_path in img_path:
            for f_name in os.listdir(dir_path):
                if f_name.endswith(img_extensions):
                    f_info.append(os.path.join(dir_path, f_name))

        return f_info
