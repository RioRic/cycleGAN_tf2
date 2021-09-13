import tensorflow as tf
from tensorflow import keras

def get_gan_loss_fn():
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = bce(tf.ones_like(real_logits), real_logits)
        fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        fake_loss = bce(tf.ones_like(fake_logits), fake_logits)
        return fake_loss
    
    return d_loss_fn, g_loss_fn

def get_hinge_v1_loss_fn():

    def d_loss_fn(real_logits, fake_logits):
        real_loss = tf.reduce_mean(tf.math.maximum(1 - real_logits), 0)
        fake_loss = tf.reduce_mean(tf.math.maximum(1 + fake_logits), 0)
        return real_loss, fake_loss

    def g_loss_fn(fake_logits):
        fake_loss = tf.reduce_mean(tf.math.maximum(1 - fake_logits), 0)
        return fake_loss
    
    return d_loss_fn, g_loss_fn

def get_lsgan_loss_fn():
    mse = keras.losses.MeanSquaredError()

    def d_loss_fn(real_logits, fake_logits):
        real_loss = mse(tf.ones_like(real_logits), real_logits)
        fake_loss = mse(tf.zeros_like(fake_logits), fake_logits)
        return real_loss, fake_loss

    def g_loss_fn(fake_logits):
        fake_loss = mse(tf.ones_like(fake_logits), fake_logits)
        return fake_loss

    return d_loss_fn, g_loss_fn

def get_wgan_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        real_loss = -tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return real_loss, fake_loss

    def g_loss_fn(fake_logits):
        fake_loss = -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn


def get_adversarial_loss(mode):
    if mode == 'gan':
        return get_gan_loss_fn()
    elif mode == 'hinge_v1':
        return get_hinge_v1_loss_fn()
    elif mode == 'lsgan':
        return get_lsgan_loss_fn()
    elif mode == 'wgan':
        return get_wgan_loss()