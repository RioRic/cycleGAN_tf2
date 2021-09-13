import tensorflow as tf

class Checkpoint():
    def __init__(self,
                 cheeckpoint_kwargs,        # for "tf.train.checkpoint"
                 directory,                 # for "tf.train.CheckpointManager"
                 max_to_keep = 5):
        self.checkpoint = tf.train.Checkpoint(**cheeckpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep)
    
    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.manager.restore(save_path)

    def save(self, checkpoint_number):
        return self.manager.save(checkpoint_number=checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return gatter(self.manager, attr)
        else:
            self.__getarrribute__(attr) # raise an exception

def summary(name_data_dict,
            step=None,
            types=['mean', 'std', 'max', 'min', 'sparsity', 'historgram'],
            histogram_buchets=None,
            name='summary'):


    def _summary(name, data):
        if data.shape == ():
            tf.summary.scalar(name, data, step=step)
        else:
            if 'mean' in types:
                tf.summary.scalar(name + '-mean', tf.math.reduce_mean(data), step=step)
            if 'std' in types:
                tf.summary.scalar(name + '-std', tf.math.reduce_std(data), step=step)
            if 'max' in types:
                tf.summary.scalar(name + '-max', tf.math.reduce_max(data), step=step)
            if 'min' in types:
                tf.summary.scalar(name + '-min', tf.math.reduce_min(data), step=step)
        
    with tf.name_scope(name):
        for name, data in name_data_dict.items():
            _summary(name, data)