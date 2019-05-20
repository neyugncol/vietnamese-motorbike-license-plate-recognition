import os
import math
import numpy as np
import cv2
import imgaug.augmenters as iaa


class DataSet(object):
    def __init__(self,
                 image_dir,
                 batch_size,
                 image_size,
                 is_train=False,
                 shuffle=True,
                 augmented=False):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.augmented = augmented
        self.setup()

    def setup(self):
        image_files = os.listdir(self.image_dir)

        self.image_files = []
        self.labels = []
        for file in image_files:
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                label = file.split('.')[0]
                if len(label) != 9:
                    continue
                self.image_files.append(file)
                self.labels.append(label)

        self.image_files = np.array(self.image_files)
        self.labels = np.array(self.labels)
        self.count = len(self.image_files)
        self.num_batches = math.ceil(self.count / self.batch_size)
        self.idxs = list(range(self.count))
        if self.augmented:
            self.build_augmentor()
        self.reset()

    def build_augmentor(self):
        self.augmentor = iaa.Sometimes(0.7,
                                       iaa.OneOf([
                                           iaa.Affine(scale=(0.8, 1.2)),
                                           iaa.FastSnowyLandscape(lightness_multiplier=2.0),
                                           iaa.Clouds(),
                                           iaa.Fog(),
                                           iaa.GammaContrast(gamma=3.0),
                                           iaa.MotionBlur(k=20),
                                           iaa.CoarseDropout(p=0.2, size_percent=1.0),
                                       ]))

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        assert self.has_next_batch()

        start = self.current_idx
        end = self.current_idx + self.batch_size
        if end > self.count:
            end = self.count
        current_idxs = self.idxs[start:end]

        image_files = self.image_files[current_idxs]

        images = self.load_images(image_files)

        labels = self.labels[current_idxs]

        return images, labels

    def has_next_batch(self):
        return self.current_idx < self.count

    def has_full_next_batch(self):
        return self.current_idx + self.batch_size <= self.count

    def load_images(self, image_files):
        images = []
        for image_file in image_files:
            image = self.load_image(self.image_dir + '/' + image_file)
            images.append(image)

        if self.augmented:
            self.augmentor.augment_images(images)

        images = np.array(images) / 255.0

        return images

    def load_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(self.image_size[:2]))

        return image
