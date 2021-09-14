import tensorflow as tf
from tensorflow import keras
from .network import ResGenerator, NLayerDiscriminator
from .loss import get_adversarial_loss

class CycleGANModel():
    def __init__(self, opt):

        self.opt = opt
        self.cycle_weight = opt.cycle_weight

        self.d_loss_fn, self.g_loss_fn = get_adversarial_loss(mode=opt.loss_mode)
                
        """
        G_A     -- A -> B 
        G_B     -- B -> A
        D_A     -- Discriminator for A, B2A
        D_B     -- Discriminator for B, A2B   
        """

        self.G_A = ResGenerator(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, 
                                opt.n_blocks, opt.padding, opt.n_downsampling)
        self.G_B = ResGenerator(opt.output_nc, opt.input_nc, opt.ngf, opt.norm,
                                opt.n_blocks, opt.padding, opt.n_downsampling)
        
        self.D_A = NLayerDiscriminator(opt.output_nc, opt.ndf, opt.padding, opt.n_layers, opt.norm)
        self.D_B = NLayerDiscriminator(opt.input_nc, opt.ndf, opt.padding, opt.n_layers, opt.norm)

        """ define optimizers """
        self.optimizer_G = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta1)
        self.optimizer_D = keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta1)

    @tf.function
    def train_step(self, A, B):
        with tf.GradientTape(persistent=True) as tape:
            """ generate fake """
            A2B = self.G_A(A, training=True)
            B2A = self.G_B(B, training=True)
            
            """ generate cycle """
            A2B2A = self.G_B(A2B, training=True)
            B2A2B = self.G_A(B2A, training=True)

            """ fake logits """
            B2A_d_logits = self.D_A(B2A, training=True)
            A2B_d_logits = self.D_B(A2B, training=True)

            """ generate loss """
            A2B_g_loss = self.g_loss_fn(A2B_d_logits)
            B2A_g_loss = self.g_loss_fn(B2A_d_logits)

            """ cycle loss """
            A2B2A_cycle_loss = self.calc_cycle_loss(A, A2B2A)
            B2A2B_cycle_loss = self.calc_cycle_loss(B, B2A2B)
            total_cycle_loss = 0.5 * (A2B2A_cycle_loss + B2A2B_cycle_loss)


            """" total loss for generator """
            total_A2B_g_loss = A2B_g_loss + total_cycle_loss
            total_B2A_g_loss = B2A_g_loss + total_cycle_loss
            G_loss = A2B_g_loss + B2A_g_loss + total_cycle_loss * self.opt.cycle_weight

            """ total loss for discriminator """
            A_d_logits = self.D_A(A)
            B_d_logits = self.D_B(B)
            A_d_loss, B2A_d_loss = self.d_loss_fn(A_d_logits, B2A_d_logits)
            B_d_loss, A2B_d_loss = self.d_loss_fn(B_d_logits, A2B_d_logits)
            D_A_loss = A_d_loss + B2A_d_loss
            D_B_loss = B_d_loss + A2B_d_loss
            D_loss = D_A_loss + D_B_loss


        G_gradients = tape.gradient(G_loss, self.G_A.trainable_variables + self.G_B.trainable_variables)
        D_gradients = tape.gradient(D_loss, self.D_A.trainable_variables + self.D_B.trainable_variables)
        self.optimizer_G.apply_gradients(zip(G_gradients, self.G_A.trainable_variables + self.G_B.trainable_variables))
        self.optimizer_D.apply_gradients(zip(D_gradients, self.D_A.trainable_variables + self.D_B.trainable_variables))
        
        g_loss_dict = {"A2B_g_loss": A2B_g_loss,
                       "B2A_g_loss": B2A_g_loss,
                       "A2B2A_cycle_loss": A2B2A_cycle_loss,
                       "B2A2B_cycle_loss": B2A2B_cycle_loss,
                       "total_A2B_g_loss": total_A2B_g_loss,
                       "total_B2A_g_loss": total_B2A_g_loss}
        d_loss_dict = {"A_d_loss": A_d_loss,
                       "B_d_loss": B_d_loss}

        return g_loss_dict, d_loss_dict

    def calc_cycle_loss(self, real_image, cycled_image):
        loss_cycle = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return loss_cycle

    def feature_loss(self):
        pass

    def sample(self, A, B):
        A2B = self.G_A(A, training=False)
        B2A = self.G_B(B, training=False)

        A2B2A = self.G_B(A2B, training=False)
        B2A2B = self.G_A(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B 
