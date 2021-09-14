import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class MyLRSchedule(LearningRateSchedule):
    def __init__(self, initial_lr):
        super(MyLRSchedule, self).__init__()
        self.initial_lr = initial_lr