import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras

def _get_norm_layer(norm):
    if norm == None:
        return lambda: lambda x : x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization()
    elif norm == 'instance_norm':
        return tfa.layers.InstanceNormalization()
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization()
    else:
        raise ValueError('Normalization not found')

class ResBlock(keras.Model):
    def __init__(self, dim, padding, norm):
        super(ResBlock, self).__init__()
        self.res_block = self.build_res_block(dim, padding, norm)

    def build_res_block(self, dim, padding, norm):
        norm_layer = _get_norm_layer(norm)
        conv_block = []
        conv_block += [
            keras.layers.Conv2D(dim, 3, padding=padding),
            norm_layer,
            keras.layers.ReLU()
        ]

        conv_block += [
            keras.layers.Conv2D(dim, 3, padding=padding),
            norm_layer
        ]
        return keras.Sequential(conv_block)

    def call(self, input):
        output = input + self.res_block(input)
        return output

class ResGenerator(keras.Model):
    def __init__(self, input_nc, output_nc, fmaps, norm="instance_norm", n_blocks=9, padding="same", n_downsampling=2):
        """ Constructe a resnet generator
        
        Parameters: 
            input_nc (int)       -- the number of channels of input images
            output_nc (int)      -- the number of channels of output images
            fmaps (int)          -- the base number of feature maps
            netG (str)           -- the name of model
            norm (str)           -- normalization layer: batch_norm | instance_norm | layer_norm
            n_blocks (int)       -- the number of resnet block
            padding (str)        -- padding type: valid | same
        """
        super(ResGenerator, self).__init__()
        assert(n_blocks >= 0)

        norm_layer = _get_norm_layer(norm)
        
        """ 1 """
        model = [
            keras.layers.Conv2D(fmaps, 7, strides=1, padding=padding),
            norm_layer,
            keras.layers.ReLU()
        ]
        """ downsampling """
        for _ in range(n_downsampling):
            fmaps *= 2
            model += [
                keras.layers.Conv2D(fmaps, 3, strides=2, padding=padding),
                norm_layer,
                keras.layers.ReLU()
            ]

        """ resblock """
        for _ in range(n_blocks):
            model += [ResBlock(fmaps, padding, norm)]

        """ upsampling """
        for _ in range(n_downsampling):
            fmaps //= 2
            model += [
                keras.layers.Conv2DTranspose(fmaps, 3, strides=2, padding=padding),
                norm_layer,
                keras.layers.ReLU()
            ]
        
        """ 2 """
        model += [
            keras.layers.Conv2D(output_nc, 7, padding=padding),
            keras.layers.Activation('tanh')
        ]
        self.model =  keras.Sequential(model)

    def call(self, input):
        return self.model(input)

class NLayerDiscriminator(keras.Model):
    """" PatchGAN Discriminator """
    def __init__(self, input_nc, fmaps, padding, n_layers=3, norm='batch_norm'):
        """ Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)      -- the number of channel of input image
            fmaps (int)         -- the number of channel of feature maps
            netD (str)           -- the name of model
            n_layers (int)      -- the number of conv
            norm (str)          -- normalization layer: batch_norm | instance_norm | layer_norm
        """
        super(NLayerDiscriminator, self).__init__()
        _fmaps = fmaps

        norm_layer = _get_norm_layer(norm)
        
        model = []
        for i in range(n_layers):
            fmaps = min(fmaps*2, _fmaps*8)
            model += [
                keras.layers.Conv2D(fmaps, 4, strides=2, padding=padding),
                norm_layer,
                keras.layers.LeakyReLU(alpha=0.2)
            ]
        
        model += [
            keras.layers.Conv2D(fmaps, 4, strides=1, padding=padding),
            norm_layer,
            keras.layers.LeakyReLU(alpha=0.2)
        ]

        model += [keras.layers.Conv2D(fmaps, 1, strides=1, padding=padding)]
        

        self.model = keras.Sequential(model)

    def call(self, input):
        return self.model(input)