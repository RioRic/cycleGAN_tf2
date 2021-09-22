import os
import datetime
import tensorflow as tf

from tqdm import tqdm
from shutil import copyfile
from model.cyclegan import CycleGANModel
from data.dataloader import DataLoader
from utils.hypermeter import Config
from utils.logger import set_logger

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def run_train(config_path):
    result_dir = 'result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = set_logger(result_dir)

    logger.info("load hypermeters from {}".format(config_path))
    config = Config(config_path)
    logger.info("Saving hypermeters in {}".format(result_dir + '/config.yaml'))
    copyfile(config_path, result_dir + '/config.yaml')

    logger.info("Loading data...")
    dataloader = DataLoader(config)
    train_data = dataloader.get_train_data()
    test_data = dataloader.get_test_data()
    test_iter = iter(test_data)

    logger.info("Build model...")
    model = CycleGANModel(config)

    logger.info("Create summary writer and checkpoint manager!")
    train_summary_writer = tf.summary.create_file_writer(result_dir + '/logs/')
    # tf.summary.trace_on(graph=True)

    cheekpoint_dir = result_dir + '/checkpoints'
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), G_A=model.G_A, G_B=model.G_B, D_A=model.D_A, D_B=model.D_B)
    manager = tf.train.CheckpointManager(ckpt, cheekpoint_dir, max_to_keep=3)
    
    logger.info("Start training!")
    with train_summary_writer.as_default():
        for epoch in tqdm(range(config.epochs)):
            logger.info("Epoch {}".format(epoch))
            for data in tqdm(train_data):
                ckpt.step.assign_add(1)
                loss_dict = model.train_step(data)
                for name, item in loss_dict.items():
                    tf.summary.scalar(name, item, step=int(ckpt.step))

                if int(ckpt.step) % config.save_epoch_freq == 0:
                    save_path = manager.save()
                    logger.info("Saved checkpoint for iteration {}: {}".format(int(ckpt.step), save_path))
                    A, B = next(test_iter)
                    A2B, B2A, A2B2A, B2A2B = model.sample(A, B)
                    A_cat = tf.concat([A, A2B, A2B2A], axis=2)
                    B_cat = tf.concat([B, B2A, B2A2B], axis=2)
                    AB = tf.concat([A_cat, B_cat], axis=1)
                    AB = (AB + 1)/2.0
                    tf.summary.image('test_{}'.format(int(ckpt.step)), AB, step=int(ckpt.step), max_outputs=10)


if __name__ == "__main__":
    run_train(config_path='./config.yaml')