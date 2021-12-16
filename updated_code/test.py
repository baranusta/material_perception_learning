from utils import *
from tf_utils import *
from model import *
import tensorflow as tf

import tensorflow.keras.preprocessing as prep
import numpy as np
import math
import cv2

from os.path import exists

def test(arg_dict):
    if(not exists(arg_dict.saved_weights_path)):
        print("weights not found ")
        return 
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model = VGG16(arg_dict)
    model.load_weights(arg_dict.saved_weights_path)

    input_files, gt_values = load_data_cvs_exclude(arg_dict.data_test_dir,arg_dict.scores_test_path, True)
    input_files = np.array(input_files[:10])
    gt_values = np.array(gt_values[:10]).astype('float32')

    length_data = len(gt_values)

    filenames_ds = tf.data.Dataset.from_tensor_slices(input_files)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    images_ds = images_ds.batch(arg_dict.batch_size)

    predictions = model.predict(images_ds, batch_size=arg_dict.batch_size)

    predictor_score = np.array(predictions)
    dataframe_gt = pd.DataFrame({
        'Name': input_files[:], 
        'glossiness(Pred)':predictor_score[:, 0], 
        'glossiness(GT)':gt_values[:, 0], 
        'refsharp(Pred)':predictor_score[:, 1], 
        'refsharp(GT)':gt_values[:, 1], 
        'contgloss(Pred)':predictor_score[:, 2], 
        'contgloss(GT)':gt_values[:, 2], 
        'metallicness(Pred)':predictor_score[:, 3], 
        'metallicness(GT)':gt_values[:, 3], 
        'lightness(Pred)':predictor_score[:, 4], 
        'lightness(GT)':gt_values[:, 4], 
        'anisotropy(Pred)':predictor_score[:, 5], 
        'anisotropy(GT)':gt_values[:, 5]})
    dataframe_gt.to_csv("predicted_score.csv", index=False,sep=',')
