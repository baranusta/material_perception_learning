from utils import *
from model import *
import tensorflow as tf
import tensorflow.keras 

import tensorflow.keras.preprocessing as prep
import numpy as np
import math
import cv2

from os.path import exists

def test(arg_dict):
    model = VGG16(arg_dict)
    if(exists(arg_dict.saved_weights_path)):
        model.load_weights(arg_dict.saved_weights_path)
        return model

def train(arg_dict):
    model = VGG16(arg_dict)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def parse_image_aug(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
        image = tf.add(image, noise)

        image = tf.image.resize(image, [arg_dict.patch_size, arg_dict.patch_size])
        return image

    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [arg_dict.patch_size, arg_dict.patch_size])
        return image
    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(arg_dict.batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds


    input_files, gt_values = load_data_cvs_exclude(arg_dict.data_train_dir,arg_dict.scores_train_path, True)
    input_files_validate, gt_values_validate = load_data_cvs_exclude(arg_dict.data_test_dir,arg_dict.scores_test_path, True)
    input_files = np.array(input_files)
    gt_values = np.array(gt_values).astype('float32')
    input_files_validate = np.array(input_files_validate)
    gt_values_validate = np.array(gt_values_validate).astype('float32')


    length_data = len(gt_values)


    filenames_ds = tf.data.Dataset.from_tensor_slices(input_files)
    images_ds = filenames_ds.map(parse_image_aug, num_parallel_calls=tf.data.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(gt_values)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)


    
    filenames_ds_val = tf.data.Dataset.from_tensor_slices(input_files_validate)
    images_ds_val = filenames_ds_val.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    labels_ds_val = tf.data.Dataset.from_tensor_slices(gt_values_validate)
    ds_val = tf.data.Dataset.zip((images_ds_val, labels_ds_val))
    ds_val = ds_val.batch(arg_dict.batch_size)
    # ds_val = configure_for_performance(ds_val)
    


    # for images, labels in ds.take(1):
    #     for i in range(3):
    #         cv2.imshow("im" + str(i), images[i].numpy())
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # show
    model.summary()

    optimizer = optimizers.Adam(arg_dict.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    model.fit(ds,
              batch_size=arg_dict.batch_size,
              epochs=arg_dict.epoch, 
              steps_per_epoch=math.ceil(length_data/arg_dict.batch_size), 
              validation_data=ds_val, 
              validation_batch_size=arg_dict.batch_size, 
              validation_steps=math.ceil(length_data/arg_dict.batch_size))
    model.save_weights(arg_dict.saved_weights_path)
    return model