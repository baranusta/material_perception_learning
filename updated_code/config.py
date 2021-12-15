from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()
config.TEST = edict()

# TRAIN ===============================================================
config.TRAIN.batch_size = 4
config.TRAIN.patch_size = 512
config.TRAIN.lr = 1e-5
config.TRAIN.beta_1 = 0.9
config.TRAIN.lr_decay = 0.8
config.TRAIN.decay_every = 20
config.TRAIN.epoch = 400
config.TRAIN.ckpt_ep = 1
config.TRAIN.log_dir = './log'
config.TRAIN.ckpt_dir = './ckpt'
config.TRAIN.re_train = False
config.TRAIN.pre_vgg = False
config.TRAIN.pretrained_model = ''
config.TRAIN.vgg16_parameters = ''

config.TRAIN.weight_decay = 0.8
config.TRAIN.learning_rate = 1e-5

# Loss coefficient
config.TRAIN.lambda_mae = 1e-3
config.TRAIN.lambda_mse = 1e-3

# Train dataset path
config.TRAIN.input_path = 'training/'
config.TRAIN.gt_path = 'scores/training_scores.csv'



##baran
config.weight_decay = 0.8
config.learning_rate = 1e-5
config.epoch = 400

config.batch_size = 4
config.patch_size = 512

config.data_train_dir = 'training/'
config.data_test_dir = 'test/'

config.scores_train_path = 'scores/training_scores.csv'
config.scores_test_path = 'scores/test_scores.csv'
config.saved_weights_path = 'weights.hdf5'
# TEST ================================================================
# Test dataset path
config.TEST.patch_size = 512
config.TEST.log_dir = './log/'
config.TEST.ckpt_dir = './weights/'
config.TEST.input_path = './test/imgs/'
config.TEST.gt_path = './test/names.csv'

ATTRB_NUM = 6