import tensorflow as tf
from .datasets import disk_image_batch_dataset
import numpy as np
import h5py
import pathlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_dataset(img_paths,
                 batch_size,
                 channels,
                 load_size,
                 crop_size,
                 training,
                 drop_remainder=True,
                 shuffle=True,
                 repeat=True,
                 ):
    all_img_paths = list(pathlib.Path(img_paths).glob('*.bmp'))
    all_img_paths = [str(path) for path in all_img_paths]
    if training:
        @tf.function
        def _map_fn(img_path):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_bmp(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255.5) / 255.0
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img_path):
            img = tf.io.read_file(img_path)
            img = tf.io.decode_bmp(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.resize(img, [crop_size, crop_size])
            img = tf.clip_by_value(img, 0, 255.0) / 255.0
            ing = img * 2 -1
            return img
    
    return disk_image_batch_dataset(all_img_paths,
                                    batch_size,
                                    drop_remainder=drop_remainder,
                                    map_fn=_map_fn,
                                    shuffle=shuffle,
                                    repeat=repeat)

def make_zip_dataset(A_image_paths, B_image_paths, batch_size, channels, load_size, crop_size, training, shuffle=True, repeat=False):
    """ repeat the datasets aligned with the longer dataset """
    
    if repeat:
        A_repeat = B_repeat = None
    else:
        if len(A_image_paths) >= len(B_image_paths):
            A_repeat = 1
            B_repeat = None
        else:
            A_repeat = None
            B_repeat = 1
    A_dataset = make_dataset(img_paths=A_image_paths, batch_size=batch_size, channels=channels,
                             load_size=load_size, crop_size=crop_size, training=training, 
                             drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(img_paths=B_image_paths, batch_size=batch_size, channels=channels,
                             load_size=load_size, crop_size=crop_size, training=training, 
                             drop_remainder=True, shuffle=shuffle, repeat=B_repeat)
        
    A_B_dataset= tf.data.Dataset.zip((A_dataset, B_dataset))

    len_dataset = max(len(A_image_paths), len(B_image_paths)) // batch_size

    return A_B_dataset, len_dataset


    
