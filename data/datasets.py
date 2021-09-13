import tensorflow as tf
import h5py
import numpy as np
import pathlib

def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  map_fn=None,
                  n_map_threads=None,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):

    
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 10, 2048)

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if map_fn:
        dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)


    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset

def disk_image_batch_dataset(img_paths,
                             batch_size,
                             drop_remainder=True,
                             n_prefetch_batch=1,
                             map_fn=None,
                             n_map_threads=None,
                             filter_after_map=False,
                             shuffle=True,
                             shuffle_buffer_size=None,
                             repeat=None):
    """ create batch dataset from disk 
    
    Parameters:

    image_path (str)            -- the path of image
    batch_size (str)            -- the size of batch
    drop_remainder (bool)       -- weather drop the last batch when 
                                   the left elements is less than batch_size
    n_prefetch_batch (int)      -- prepare the later elements when processing
                                   current elements
    map_fn (function)           -- the map function
    shuffle (bool)              -- weather shuffle
    shuffle_buffer_size (int)   -- the size of shuffle
    repeat (bool)               -- weather repeat

    Return:
    
    batch dataset for train or test
    """

    img_paths = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = batch_dataset(img_paths,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset

