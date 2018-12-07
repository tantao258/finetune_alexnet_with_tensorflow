import tensorflow as tf
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

class ImageDataGenerator(object):
    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                  different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data
                in the dataset and the initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """

        img_paths = []
        labels = []
        with open(txt_file, 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                img_paths.append(items[0])
                labels.append(int(items[1]))

        data_size = len(labels)

        if shuffle:
            permutation = np.random.permutation(data_size)
            img_paths_ = []
            labels = []
            for i in permutation:
                self.img_paths.append(path[i])
                self.labels.append(labels[i])
