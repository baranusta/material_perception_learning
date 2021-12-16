
from config import *
import tensorflow as tf

def parse_image_aug(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    image = tf.add(image, noise)

    image = tf.image.resize(image, [config.patch_size, config.patch_size])
    return image

def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [config.patch_size, config.patch_size])
    return image

def configure_for_performance(ds):
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(config.batch_size)
    ds = ds.repeat()
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds
