# coding=utf-8
import tensorflow as tf
import numpy as np
import re


# writer=tf.python_io.TFRecordWriter('') Piclength=5, Picwidth=5, Picdimension=5,
def read_and_decode(filename_queue,imageshape,labelnum):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'feature': tf.FixedLenFeature([], tf.string),

            'label': tf.FixedLenFeature([],tf.string)
        })
    image = tf.decode_raw(features['feature'], tf.float64)
    image = tf.reshape(image,imageshape)
    label =tf.decode_raw(features['label'], tf.float64)
    label= tf.reshape(label,[labelnum])
    return image, label


tfrecord_filename = 'D2_traindata(5_7)/data-test.tfrecord'
imageshape=re.findall(r"\d+\.?\d*",tfrecord_filename)
imageshape=(5,5,2)
# test_data
tfrecord_filename_test = 'D2_traindata(5_7)/data-test.tfrecord'
tfrecord_filename_val='D2_traindata(5_7)/data-val.tfrecord'
labelnum=6
filename_queue = tf.train.string_input_producer([tfrecord_filename_test], )
image, label = read_and_decode(filename_queue, imageshape, labelnum)
image_batch, label_batch = tf.train.batch([image, label], batch_size=1000, capacity=5000)
# val data
filename_queue_val = tf.train.string_input_producer([tfrecord_filename_val], )
image_val, label_val = read_and_decode(filename_queue_val, imageshape, labelnum)
image_batch_val, label_batch_val = tf.train.batch([image_val, label_val], batch_size=100, capacity=500)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    merged = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10000):
        imagebatch, labelbatch, imagebatchval, labelbatchval = sess.run(
            [image_batch, label_batch, image_batch_val, label_batch_val])
        if i % 100 == 0:
            print(i,imagebatch.shape, labelbatch.shape, imagebatchval.shape, labelbatchval.shape)

    coord.request_stop()
    coord.join(threads)

