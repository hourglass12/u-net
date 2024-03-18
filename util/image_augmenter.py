import random
import tensorflow as tf
import numpy as np
from util import loader as ld

class ImageAugmenter:
    NONE = 0
    FLIP = 1
    BRIGHTNESS = 2
    HUE = 3
    SATURATION = 4

    NUMBER_OF_AUGMENT = 5

    def __init__(self, size, class_count):
        self._class_count = class_count
        self._width, self._height = size[0], size[1]
        self.init_graph()

    def augment_dataset(self, dataset, method=None):
        input_processed = []
        output_processed = []
        for ori, seg in zip(dataset.images_original, dataset.images_segmented):
            ori_processed, seg_processed = self.augment(ori, seg, method)
            input_processed.append(ori_processed)
            output_processed.append(seg_processed)

        return ld.DataSet(np.asarray(input_processed), np.asarray(output_processed), dataset.palette)

    def augment(self, image_in, image_out, method=None):
        if method is None:
            idx = random.randrange(ImageAugmenter.NUMBER_OF_AUGMENT)
        else:
            assert len(method) <= ImageAugmenter.NUMBER_OF_AUGMENT, "method is too many."
            if ImageAugmenter.NONE not in method:
                method.append(ImageAugmenter.NONE)
            idx = random.choice(method)

        op = self._operation[idx]
        return op(image_in, image_out)

    def init_graph(self):
        self._operation = {
            ImageAugmenter.NONE: lambda x, y: (x, y),
            ImageAugmenter.FLIP: self.flip,
            ImageAugmenter.BRIGHTNESS: self.brightness,
            ImageAugmenter.HUE: self.hue,
            ImageAugmenter.SATURATION: self.saturation
        }

    def flip(self, image_in, image_out):
        image_out_index = tf.argmax(image_out, axis=2)
        image_out_index = tf.expand_dims(image_out_index, axis=-1)
        image_in_processed = tf.image.flip_left_right(image_in)
        image_out_processed = tf.image.flip_left_right(tf.one_hot(image_out_index, depth=self._class_count))
        return image_in_processed, image_out_processed

    def brightness(self, image_in, image_out):
        max_delta = 0.3
        image_in_processed = tf.image.random_brightness(image_in, max_delta)
        return image_in_processed, image_out

    def hue(self, image_in, image_out):
        max_delta = 0.5
        image_in_processed = tf.image.random_hue(image_in, max_delta)
        return image_in_processed, image_out

    def saturation(self, image_in, image_out):
        lower, upper = 0.0, 1.2
        image_in_processed = tf.image.random_saturation(image_in, lower, upper)
        return image_in_processed, image_out

if __name__ == "__main__":
    pass
