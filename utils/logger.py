import os
import sys
import logging


def set_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        raise SystemExit()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    """ print log message with INFO level or above onto the screen """
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if os.path.exists(work_dir):
        raise SystemExit('Work diextory {} has already exists!'.format(work_dir))
    os.makedirs(work_dir)

    """ save log message with all level"""
    fh = logging.FileHandler(os.path.join(work_dir, logfile_name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger