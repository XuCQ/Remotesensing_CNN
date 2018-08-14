# coding=utf-8
import tensorflow as tf
import os
import math
import re
import numpy as np
import time

def computer_accuracy(v_xs,v_ys):
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    with tf.name_scope('accuracy'):
        correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1)) #tf.argmax(a,dimension),返回的是a中的某个维度最大值的索引;tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    with tf.name_scope('accuracy'):
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #cast(x, dtype, name=None) 将x的数据格式转化成dtype,求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result


def weight_variable(shape,name='weights'):
    """
    :param shape: The shape of weight
    :return: weight
    """
    shape=[np.int32(i) for i in shape]
    inital = tf.truncated_normal(shape,stddev=0.1)  # tf.truncated_normal(shape, mean, stddev) :从截断的正态分布中输出随机值;shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。
    return tf.Variable(inital,name=name)


def bias_variable(shape,name='biases'):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital,name=name)


def conv2d(x, W, x_movement=1, y_movement=1, name='conv2d'):
    return tf.nn.conv2d(x, W, strides=[1, x_movement, y_movement, 1], padding='SAME',name=name)


def max_pool_2x2(x,name='max_pool_2x2'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

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
    image = tf.reshape(image,[ imageshape[0] * imageshape[1] * imageshape[2]])
    label =tf.decode_raw(features['label'], tf.float64)
    label= tf.reshape(label,[labelnum])
    return image, label

# 参数设定
def trainmodel(tfrecord_filename_test,tfrecord_filename_val,PicShape, labelnum):
    global xs, ys, keep_prob, sess,prediction
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, PicShape[0] * PicShape[1] * PicShape[2]])
        ys = tf.placeholder(tf.float32, [None, labelnum])
        keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(xs, [-1, PicShape[0], PicShape[1], PicShape[2]])
    # ==========CNN begin================
    # ==========conv layer begin================
    # conv0 layer no normailization ,but no pool
    with tf.name_scope('conv0'):
        W_conv0 = weight_variable(
            [3, 3, PicShape[2], PicShape[2]])  # shape [3,3,7,7] insize 7 outsize 7 and the shapr of filter is [3,3,7]
        b_conv0 = bias_variable([PicShape[2]])
        h_conv0 = conv2d(x_image, W_conv0, 1, 1) + b_conv0
        tf.summary.histogram('conv0/outputs', h_conv0)

    # conv1 layer with normailization ,but no pool
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, PicShape[2], PicShape[2] * math.pow(2,1)])  # shape:insize 7 outsize 14, and the shapr of filter is [3,3,14]
        b_conv1 = bias_variable([PicShape[2] * math.pow(2, 1)])
        h_conv1 = tf.nn.relu(conv2d(h_conv0, W_conv1, 1, 1) + b_conv1)
        tf.summary.histogram('conv1/outputs', h_conv1)

    # conv2 layer no normailization, but no pool
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, PicShape[2] * math.pow(2, 1), PicShape[2] * math.pow(2, 2)])  # shape: insize 14, outsize: 28, and the shape of filter is [3,3,28]
        b_conv2 = bias_variable([PicShape[2] * math.pow(2, 2)])
        h_conv2 = conv2d(h_conv1, W_conv2, 1, 1) + b_conv2
        tf.summary.histogram('conv2/outputs', h_conv2)

    # conv3 layer with normailzation ,but no pool
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([3, 3, PicShape[2] * math.pow(2, 2), PicShape[2] * math.pow(2,3)])  # shape: insize 28, outsize 56. And the shape of filter is [3,3,56]
        b_conv3 = weight_variable([PicShape[2] * math.pow(2, 3)])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1, 1) + b_conv3)
        tf.summary.histogram('conv3/outputs', h_conv3)

    # conv4 layer with normailization, but no pool
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable([3, 3, PicShape[2] * math.pow(2, 3), PicShape[2] * math.pow(2,4)])  # shape :insize 56, outsize 112. And the shape of filter is [3,3,112]
        b_conv4 = weight_variable([PicShape[2] * math.pow(2, 4)])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
        tf.summary.histogram('conv4/outputs', h_conv4)
    # ==========conv layer end================
    # ==========func layer end================
    #func0 layer
    with tf.name_scope('func0'):
        W_fc0=weight_variable([PicShape[0]*PicShape[1]*PicShape[2] * math.pow(2,4),PicShape[0]*PicShape[1]*PicShape[2] * math.pow(2,4)])   #function layer shape:insize PICsize 5*5*112 outsize5*5*112
        b_fc0=bias_variable([PicShape[0]*PicShape[1]*PicShape[2] * math.pow(2,4)])
        h_flat=tf.reshape(h_conv4,[-1,PicShape[0]*PicShape[1]*np.int32(PicShape[2] * math.pow(2,4))])
        h_fc0=tf.nn.relu(tf.matmul(h_flat,W_fc0)+b_fc0)
        h_fc0_drop = tf.nn.dropout(h_fc0, keep_prob)    #avoid overfitting
        tf.summary.histogram('func0/outputs', h_fc0_drop)
    #func1 layer
    with tf.name_scope('func1'):
        W_fc1=weight_variable([PicShape[0]*PicShape[1]*PicShape[2] * math.pow(2,4),labelnum])
        b_fc1=bias_variable([labelnum])
        h_fc1=tf.matmul(h_fc0_drop,W_fc1)+b_fc1
        tf.summary.histogram('func1/outputs', h_fc1)
    with tf.name_scope('cost'):
        prediction=tf.nn.softmax(h_fc1,name='prediction')     #求取输出属于某一类的概率
        tf.summary.scalar('prediction_min', tf.reduce_min(prediction))
        prediction=tf.clip_by_value(prediction,1e-8,tf.reduce_max(prediction))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))      #交叉熵
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        # 1e-4
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        tf.summary.histogram('train/gradients', train_step)
    with tf.name_scope('accuracy'):
        correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1)) #tf.argmax(a,dimension),返回的是a中的某个维度最大值的索引;tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #cast(x, dtype, name=None) 将x的数据格式转化成dtype,求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
        tf.summary.scalar('accuracy', accuracy)
    # ==========func layer end================
    # ==========train data begin================
    #test_data
    filename_queue = tf.train.string_input_producer([tfrecord_filename_test], )
    image, label = read_and_decode(filename_queue, imageshape,labelnum)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=1000, capacity=5000)
    #val data
    filename_queue_val = tf.train.string_input_producer([tfrecord_filename_val], )
    image_val, label_val = read_and_decode(filename_queue_val, imageshape,labelnum)
    image_batch_val, label_batch_val = tf.train.batch([image_val, label_val], batch_size=100, capacity=500)
    # image_batch=tf.cast(image_batch, tf.float32, name=None)
    # label_batch = tf.cast(label_batch, tf.float32, name=None)
    # ==========train data end================
    # ==========train begin================
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("D2_log"+ str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))), sess.graph)
        saver=tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10000):
            imagebatch,labelbatch,imagebatchval,labelbatchval=sess.run([image_batch,label_batch,image_batch_val, label_batch_val])
            sess.run(train_step, feed_dict={xs: imagebatch, ys: labelbatch, keep_prob: 0.8})
            if i % 100 == 0:
                result_test=computer_accuracy(imagebatch, labelbatch)
                # result_val=computer_accuracy(imagebatchval,labelbatchval)
                summary, acc = sess.run([merged, accuracy], feed_dict={xs: imagebatchval, ys: labelbatchval, keep_prob: 1})
                writer.add_summary(summary, i)
                print('Accuracy at step %s:val: %s; test:%s' % (i, acc,result_test))
            else:
                summary, _ = sess.run([merged, train_step], feed_dict={xs: imagebatch, ys: labelbatch, keep_prob: 0.8})
                writer.add_summary(summary, i)

        coord.request_stop()
        coord.join(threads)

tfrecord_filename_test = 'D2_traindata(5_7)/data-test.tfrecord'
tfrecord_filename_val='D2_traindata(5_7)/data-val.tfrecord'
# imageshape=re.findall(r"\d+\.?\d*",tfrecord_filename)
imageshape=(5,5,2)
trainmodel(tfrecord_filename_test,tfrecord_filename_val,imageshape,6)