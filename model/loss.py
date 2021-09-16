import tensorflow as tf
from tensorflow import keras
""" diffierent gan loss """
def get_gan_loss_fn():
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = bce(tf.ones_like(real_logits), real_logits)
        fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
        return real_loss, fake_loss

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

def get_hinge_v2_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        real_loss = tf.reduce_mean(tf.math.maximum(1 - real_logits), 0)
        fake_loss = tf.reduce_mean(tf.math.maximum(1 + fake_logits), 0)
        return real_los, fake_loss

    def g_loss_fn(fake_logits):
        fake_loss = tf.reduce_mean(tf.math.maximum( - fake_logits), 0)
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

def gradient_penalty(f, real, fake, mode):
    def _gradient_penalty(f, real, fake=None):
        def _interpolate(a, b=None):
            if b is None:
                beta = tf.random.uniform(shape=tf.shape(a), minval=0., maxval=1.)
                b = a + 0.5 * tf.math,reduce_std(a) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter
        
        x = _interpolate(real, fake)
        with tf.gradientTape() as t:
            t.watch(x)
            pred = f(x)
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.0)**2)
        return gp
    
    if mode is None:
        gp = tf.constant(0, dtype=real.dtype)
    elif mode == 'dragan':
        gp = _gradient_penalty(f, real)
    elif mode =='wgan-gp':
        gp = _gradient_penalty(f, real, fake)
    return gp
