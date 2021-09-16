import tensorflow as tf
import numpy as np
import pathlib
import os

class DataLoader(object):
    def __init__(self, config):
        self.datapath = config.datapath
        
        self.buffer_size = config.buffer_size
        self.train_batch_size = config.train_batch_size
        self.test_batch_size = config.test_batch_size

        self.resize_size = config.resize_size
        self.crop_size = config.crop_size
        self.channel = config.channel

    def map(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_bmp(image)
        image = tf.image.resize(image, [self.resize_size, self.resize_size])
        cropped_image = tf.image.random_crop(image, size=[self.crop_size, self.crop_size, self.channel])
        mirror_image = tf.image.random_flip_left_right(cropped_image)
        image = tf.cast(mirror_image, tf.float32)
        image = image / 255.0 * 2 -1
        return image

    def load_image_paths(self, path):
        all_image_paths = list(pathlib.Path(path).glob('*.bmp'))
        all_image_paths = [str(path) for path in all_image_paths]
        return all_image_paths

    def make_dataset(self, paths, repeat, training):
        image_paths = tf.data.Dataset.from_tensor_slices(paths)
        image_paths = image_paths.shuffle(self.buffer_size)
        dataset = image_paths.map(self.map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if training:
            dataset = dataset.batch(self.train_batch_size, drop_remainder=True)
        else:
            dataset = dataset.batch(self.test_batch_size, drop_remainder=True)
        dataset = dataset.repeat(repeat).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def make_zip_dataset(self, A_paths, B_paths, repeat, training):
        if repeat:
            repeat_A = repeat_B = None
        else:
            if len(A_paths) > len(B_paths):
                repeat_A = 1
                repeat_B = None
            else:
                repeat_A = None
                repeat_B = 1
        dataset_A = self.make_dataset(A_paths, repeat=repeat_A, training=training)
        dataset_B = self.make_dataset(B_paths, repeat=repeat_B, training=training)

        A_B_dataset= tf.data.Dataset.zip((dataset_A, dataset_B))

        return A_B_dataset

    def get_train_data(self):
        train_A_path = self.datapath + '/train_A/'
        train_B_path = self.datapath + '/train_B/'
        train_A_paths = self.load_image_paths(train_A_path)
        train_B_paths = self.load_image_paths(train_B_path)
        print(len(train_A_paths))
        return self.make_zip_dataset(train_A_paths, train_B_paths, repeat=False, training=True)

    def get_test_data(self):
        test_A_path = self.datapath + '/test_A/'
        test_B_path = self.datapath + '/test_B/'
        test_A_paths = self.load_image_paths(test_A_path)
        test_B_paths = self.load_image_paths(test_B_path)
        return self.make_zip_dataset(test_A_paths, test_B_paths, repeat=True, training=False)