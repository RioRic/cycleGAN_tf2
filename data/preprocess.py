import tensorflow as tf


def get_map_fn():

@tf.function
def decode_bmp_fn(img):
    img = tf.io.read_file(img)
    img =tf.io.decode_bmp(img)

@
