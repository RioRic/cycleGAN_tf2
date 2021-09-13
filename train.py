import tensorflow as tf
import json
from model.cyclegan import CycleGANModel
from data.dataloader import make_zip_dataset
from utils.hypermeter import Config
from utils.checkpoint import Checkpoint, summary 
import argparse
import os
from shutil import copyfile
import numpy as np
from PIL import Image
import datetime
from utils.logger import set_logger

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--A_path', type=str, required=True)
    parser.add_argument('--B_path', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)

    return parser.parse_args()

def main():
    # args = parse_arg()
    result_dir= 'result/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger = set_logger(work_dir=result_dir)
    
    logger.info("load the hypermeters from {}".format(config))
    config = Config('/opt/home/guangjiel/cyclegan/config.json')

    logger.info('load and preprocess datasets for train and test')
    A_path = '/mnt/nfs/share/guangjiel/data/dataset/cycleGAN/target'
    B_path = '/mnt/nfs/share/guangjiel/data/dataset/cycleGAN/probe'
    datasets, size = make_zip_dataset(A_image_paths=A_path, 
                                B_image_paths=B_path, 
                                batch_size=config.batch_size, 
                                load_size=config.load_size, 
                                crop_size=config.crop_size, 
                                training=True, 
                                shuffle=True, 
                                repeat=False)
    A_path = '/mnt/nfs/share/guangjiel/data/dataset/cycleGAN/target_test'
    B_path = '/mnt/nfs/share/guangjiel/data/dataset/cycleGAN/probe_test'
    test_dataset, _ = make_zip_dataset(A_image_paths=A_path, 
                                    B_image_paths=B_path, 
                                    batch_size=1, 
                                    load_size=640, 
                                    crop_size=640, 
                                    training=False, 
                                    shuffle=True, 
                                    repeat=None)
    test_iter = iter(test_dataset)

    logger.info("Builing model!")
    model = CycleGANModel(config)

    iteration = 0
    train_summary_writer = tf.summary.create_file_writer(os.path.join(result_dir, 'summaries'))
    with train_summary_writer.as_default():
        for i in range(config.epochs):
            logger.info("")
            for A_batch, B_batch in datasets:
                iteration += 1
                g_loss_dict, d_loss_dict = model.train_step(A_batch, B_batch)
                logger.info("train iteration {}".format(iteration))
                summary(g_loss_dict, step=iteration, name="G_losses")
                summary(d_loss_dict, step=iteration, name="D_losses")
                if iteration % 100 == 0:
                    A, B = next(test_iter)
                    A2B, B2A, A2B2A, B2A2B = model.sample(A, B)
                    image_A = np.concatenate([np.squeeze(A), np.squeeze(A2B), np.squeeze(A2B2A)], axis=1)
                    image_B = np.concatenate([np.squeeze(B), np.squeeze(B2A), np.squeeze(B2A2B)], axis=1)
                    image = np.concatenate([image_A, image_B], axis=0)
                    image = (image + 1) / 2.0 * 255.0
                    image = np.clip(image, 0, 255)
                    image = Image.fromarray(image.astype(np.uint8))
                    image.save('{}/iteration_{}'.format(result_dir, iteration) + '.bmp')


if __name__ == "__main__":
    main()