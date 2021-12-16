from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()
##baran
config.weight_decay = 0.8
config.learning_rate = 1e-5
config.epoch = 10

config.batch_size = 4
config.patch_size = 512

config.data_train_dir = 'training/'
config.data_test_dir = 'test/'

config.scores_train_path = 'scores/training_scores.csv'
config.scores_test_path = 'scores/test_scores.csv'
config.saved_weights_path = 'weights.hdf5'

ATTRB_NUM = 6