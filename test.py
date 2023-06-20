import tensorflow as tf
import os
from model import ZXY_Net
from dataset_tfrecord import get_dataset_singCoil
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary, mse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', metavar='str', nargs=1, default=['test'], help='training or test')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['16'], help='accelerate rate')
    parser.add_argument('--net', metavar='str', nargs=1, default=['SLRNet'], help='SLR Net or S Net')
    parser.add_argument('--weight', metavar='str', nargs=1, default=['models/stable/2023-04-18T08-12-40ZXY_NET_SIRST12_lr_0.001/epoch-30/ckpt'], help='modeldir in ./models')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--data', metavar='str', nargs=1, default=['DYNAMIC_V2_MULTICOIL'], help='dataset name')
    parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    dataset_name = 'SIRST'
    mode = args.mode[0]
    batch_size = int(args.batch_size[0])
    niter = int(args.niter[0])
    acc = int(args.acc[0])
    net_name = args.net[0].upper()
    weight_file = args.weight[0]
    learnedSVT = bool(args.learnedSVT[0])

    print('network: ', net_name)
    print('acc: ', acc)
    print('load weight file from: ', weight_file)


    result_dir = os.path.join('results/stable', weight_file.split('/')[2] + net_name + str(acc) + '_lr_0.001')
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, TIMESTAMP + net_name + str(acc) + '/'))

    # prepare dataset
    dataset = get_dataset_singCoil(batch_size, shuffle=False)
    
    # initialize network

    net = ZXY_Net(niter, learnedSVT)
        

    net.load_weights(weight_file)
    
    # Iterate over epochs.
    for i, sample in enumerate(dataset):
        # forward
        csm = None
        #with tf.GradientTape() as tape:
        TrainData, TrainLabel = sample

        TrainData = TrainData.numpy()
        Tmax = TrainData.max()
        Tmin = TrainData.min()
        TrainData = (TrainData - Tmin) / (Tmax - Tmin)
        TrainData = tf.Variable(TrainData)

        TrainLabel = tf.Variable(TrainLabel)
        TrainLabel = tf.where(TrainLabel.numpy() < 2, 0., 1.)
        label_abs = tf.abs(TrainLabel)

        
        t0 = time.time()
        T, T_SYM, B = net(TrainData, TrainLabel)
        t1 = time.time()
    
        recon_abs = tf.abs(T)
        loss_total = tf.reduce_mean(tf.multiply(tf.square(T - TrainLabel), 1 - TrainLabel))
        # loss_total = mse(T, TrainLabel)

        tf.print(i, 'mse =', loss_total.numpy(), 'time = ', t1-t0)

        result_file = os.path.join(result_dir, 'T_'+str(i+1)+'.mat')
        
        datadict = {'T': np.squeeze(tf.transpose(T, [0,1]).numpy())}
        scio.savemat(result_file, datadict)

        result_file = os.path.join(result_dir, 'B_' + str(i + 1) + '.mat')
        B = tf.reshape(B, (676,2500))
        datadict = {'B': np.squeeze(tf.transpose(B, [0, 1]).numpy())}
        scio.savemat(result_file, datadict)


        TrainLabel = tf.reshape(TrainLabel, (676, 2500))
        result_file = os.path.join(result_dir, 'Label_' + str(i + 1) + '.mat')
        datadict = {'L': np.squeeze(tf.transpose(TrainLabel, [0, 1]).numpy())}
        scio.savemat(result_file, datadict)

        TrainData = tf.reshape(TrainData, (676, 2500))
        result_file = os.path.join(result_dir, 'DATA_' + str(i + 1) + '.mat')
        datadict = {'D': np.squeeze(tf.transpose(TrainData, [0, 1]).numpy())}
        scio.savemat(result_file, datadict)

        # record gif


