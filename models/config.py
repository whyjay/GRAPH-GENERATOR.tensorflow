import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

from models.build_model import build_model
from models.train import train
from models.generator import *
#from models.evaluate import evaluate
from utils import pp, visualize, to_json

from IPython import embed

class Config(object):
    def __init__(self, FLAGS):
        timestamp = str(int(time.time()))
        self.dataset = FLAGS.dataset
        self.dataset_path = os.path.join("/data2/whyjay/sna_data", self.dataset)
        self.devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]

        self.add_noise = True
        self.noise_stddev = 0.1

        timestamp = str(int(time.time()))

        self.epoch = FLAGS.epoch
        self.log_dir = os.path.join('logs', timestamp)
        self.checkpoint_dir = os.path.join('checkpoint', timestamp)
        self.sample_dir = os.path.join('samples', timestamp)
        self.timestamp = timestamp

        self.encoder_name = 'base_g_xz'
        self.decoder_name = 'base_g_zx'
        self.encoder_func = globals()[self.encoder_name]
        self.decoder_func = globals()[self.decoder_name]

        self.build_model_func = build_model
        self.train_func=train

        self.loss = FLAGS.loss

        # Learning rate
        self.learning_rate=1e-3

        self.batch_size=64
        self.y_dim=1
        self.image_size=128
        self.image_shape=[128, 128, 1]
        self.f_dim=1
        self.c_dim=1
        self.z_dim=256 # 256, 10

        self.sample_size=1*self.batch_size

    def print_config(self):
        dicts = self.__dict__
        for key in dicts.keys():
            print key, dicts[key]

    def make_dirs(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
