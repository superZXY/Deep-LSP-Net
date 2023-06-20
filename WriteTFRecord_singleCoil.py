import mat73
import scipy.io
import tensorflow as tf
import glob
import os
import numpy as np


def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 创建图像数据的Example
def data_example(data_dir):

    dataset = scipy.io.loadmat(data_dir)
    data = dataset['data']
    label = dataset['label']
    #data = np.array(data['data'])
    #data = np.transpose(data, (1, 0))  # nx, ny, nt -> nt, nx, ny

    data = np.array(data)
  #  max_data = np.max(np.abs(data[:]))
    #data = tf.constant(data / max_data)


    data = np.array(data)
    label = np.array(label)

    data_shape = data.shape
    data = data.flatten()

    label_shape = label.shape
    label = label.flatten()

    feature = {
        'data': _float_feature(data.tolist()),
        'label': _float_feature(label.tolist()),
        'data_shape': _int64_feature(list(data_shape))
    }

    exam = tf.train.Example(features=tf.train.Features(feature=feature))

    return exam


writer = tf.io.TFRecordWriter('test7_SIRST.tfrecord')
data_dirs = glob.glob(os.path.join('./IPI-for-small-target-detection-master/mat/7/', '*.mat'))
for data_dir in data_dirs:
    print(data_dir)
    exam = data_example(data_dir)
    writer.write(exam.SerializeToString())
writer.close()



