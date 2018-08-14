# 参数添加NDVI NDBI
# 参数添加MNDWI
# 添加PCA模块
import numpy as np
import math
import scipy
from libtiff import TIFF, TIFFimage
from scipy import signal
import os
import tensorflow as tf
import random
import progressbar
from sklearn.decomposition import PCA


def label_rename(PICsize, label):
    if PICsize == 5:
        if label == 1 or label == 3 or label == 21:
            return 1
        elif label == 4 or label == 12 or label == 22 or label == 24 or label == 25:
            return 2
        elif label == 18 or label == 26:
            return 3
        elif label == 10 or label == 19:
            return 4
        elif label == 5:
            return 5
        elif label == 8 or label == 13:
            return 6
        else:
            # print('label分类出错',label)
            return -1
    else:
        print('PICsize暂未支持', PICsize)
        return label


def labellist(label, labelmax):
    label_list = np.zeros(shape=labelmax)
    label_list[label - 1] = 1
    return label_list


def getNDVI_NDBI(NIR, RED, MIR):
    NDVI = (NIR - RED) / (NIR + RED)
    NDBI = (MIR - NIR) / (MIR + NIR)
    return NDVI, NDBI


def getMNDWI(Green, MIR):
    MNDWI = (Green - MIR) / (Green + MIR)
    return MNDWI


def PCA_process(data, n_components):
    datashape = data.shape
    data = np.reshape(data, (-1, datashape[-1]))
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)
    return np.reshape(data, datashape)


def find_same_pixel(originalPIC, PICsize, label_dimension=7, ignore_pixel=256, same_rate=1):
    """
    :param originalPIC:
    :param PICsize: The size of the cut image, just support int
    :param label_dimension:
    :param ignore_pixel: 1-256
    :param same_rate:
    :return:simple_pics,simple_pics
    """
    originalPIC_zAisx = originalPIC.shape[-1]
    originalPIC_label = originalPIC[:, :, label_dimension]
    # print(list(originalPIC))
    padding_piexl = math.ceil(PICsize / 2)
    originalPIC_long, originalPIC_width = originalPIC_label.shape
    indexs = np.argwhere(originalPIC_label == ignore_pixel)
    for index in indexs:
        originalPIC_label[index[0]][index[1]] = 0
    # print(list(originalPIC))
    pixel_xrange = (padding_piexl, originalPIC_long - padding_piexl)
    pixel_yrange = (padding_piexl, originalPIC_width - padding_piexl)

    scharr = np.ones((PICsize, PICsize))
    grad = signal.convolve2d(originalPIC_label, scharr, boundary='fill', mode='same')  # 边缘补0
    pic_nun = 1
    simple_pics = []
    simple_labels = []
    # print('样本信息提取   进度：1/2')
    with progressbar.ProgressBar(min_value=pixel_xrange[0], max_value=pixel_xrange[1]) as bar:
        for x_conv in range(pixel_xrange[0], pixel_xrange[1]):
            bar.update(x_conv)
            for y_conv in range(pixel_yrange[0], pixel_yrange[1]):
                if grad[x_conv][y_conv] == 0:
                    continue
                else:
                    # 判断卷积是否符合指标
                    if grad[x_conv][y_conv] == originalPIC_label[x_conv][y_conv] * math.pow(PICsize, 2):
                        # 判断卷积开始位置
                        if (PICsize % 2) == 0:  # 偶数
                            x_position, y_position = x_conv - int(PICsize / 2), y_conv - int(PICsize / 2)
                        else:  # 奇数
                            x_position, y_position = x_conv - math.floor(PICsize / 2), y_conv - math.floor(PICsize / 2)
                        # simple_pic = np.zeros((PICsize, PICsize, originalPIC_zAisx - 1))
                        simple_pic = np.zeros((PICsize, PICsize, 2))
                        simple_pic_x = 0
                        simple_pic_y = 0
                        simple_pic_judge = False
                        for xconv_out in range(x_position, x_position + PICsize):
                            simple_pic_y = 0
                            if simple_pic_judge:
                                break
                            for yconv_out in range(y_position, y_position + PICsize):
                                if originalPIC[x_conv][y_conv][label_dimension] != originalPIC[xconv_out][yconv_out][
                                    label_dimension]:
                                    simple_pic_judge = True
                                    break
                                for DN in originalPIC[xconv_out][yconv_out][0:label_dimension]:
                                    if int(DN) < 0:
                                        simple_pic_judge = True
                                        break
                                NDVI, NDBI = getNDVI_NDBI(originalPIC[xconv_out][yconv_out][5 - 1],
                                                          originalPIC[xconv_out][yconv_out][4 - 1],
                                                          originalPIC[xconv_out][yconv_out][7 - 1])
                                # simple_pic[simple_pic_x][simple_pic_y] = originalPIC[xconv_out][yconv_out][0:label_dimension]
                                # 添加MNDWI
                                MNDWI = getMNDWI(originalPIC[xconv_out][yconv_out][3 - 1],
                                                 originalPIC[xconv_out][yconv_out][7 - 1])
                                simple_pic[simple_pic_x][simple_pic_y] = [NDVI, NDBI]
                                simple_pic_y += 1
                            simple_pic_x += 1
                        if simple_pic_judge:
                            continue
                        label = label_rename(PICsize, int(originalPIC[x_conv][y_conv][label_dimension]))
                        if label == -1:
                            continue
                        else:
                            simple_pics.append(simple_pic)
                            simple_labels.append(label)
                            pic_nun += 1
                    else:
                        continue
    print('数据量', pic_nun)
    return simple_pics, simple_labels


def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    print(feature.shape)
    print(label.shape)
    tfrecords_features['feature'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
    tfrecords_features['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


# 把所有数据写入tfrecord文件
def make_tfrecord(data, outf_nm='data-train'):
    feats, labels = data
    outf_nm += '.tfrecord'
    tfrecord_wrt = tf.python_io.TFRecordWriter(outf_nm)
    ndatas = len(labels)
    # print('样本信息存储   进度：2/2')
    with progressbar.ProgressBar(max_value=nDatas) as bar:
        for inx in range(ndatas):
            exmp = get_tfrecords_example(feats[inx], labels[inx])
            exmp_serial = exmp.SerializeToString()
            tfrecord_wrt.write(exmp_serial)
            bar.update(inx)
    tfrecord_wrt.close()


def _parse_function(filename):
    image_string = TIFF.open(filename, mode='r')
    image_decoded = image_string.read_image()
    return image_decoded


def make_dir(savepath):
    folder = os.getcwd() + "\\" + savepath
    print(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)


PIC_size = 5
PIC_dimension = 7
tif = TIFF.open('./data_RS/c20140316LC8_CompositeBands1/c20140316LC8_CompositeBands1.tif', mode='r')
ar = tif.read_image()
simple_pics, simple_labels = find_same_pixel(ar, PIC_size, PIC_dimension)
labelmax = max(simple_labels)
simple_labels = [labellist(label, labelmax) for label in simple_labels]
nDatas = len(simple_pics)
inx_lst = [i for i in range(nDatas)]
random.shuffle(inx_lst)
random.shuffle(inx_lst)
ntrains = int(0.85 * nDatas)
savedir = 'D2_traindatatext(' + str(PIC_size) + '_' + str(PIC_dimension) + ')'
make_dir(savedir)
# make training set
data = ([simple_pics[i] for i in inx_lst[:ntrains]], [simple_labels[i] for i in inx_lst[:ntrains]])
make_tfrecord(data, outf_nm=savedir + '/data-train')
# make validation set
data = ([simple_pics[i] for i in inx_lst[ntrains:]], [simple_labels[i] for i in inx_lst[ntrains:]])
make_tfrecord(data, outf_nm=savedir + '/data-val')
# make test set
data = (simple_pics, simple_labels)
make_tfrecord(data, outf_nm=savedir + '/data-test')
