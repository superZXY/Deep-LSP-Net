import tensorflow as tf
import os
from model import ZXY_Net
from dataset_tfrecord import get_dataset
from dataset_tfrecord import get_dataset_singCoil
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools.tools import video_summary

from tools.tools import mse, loss_function_ISTA
from einops import rearrange

#tf.debugging.set_log_device_placement(True)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":
    '''
    layer = tf.keras.layers.Dense(2, activation='relu')
    x = tf.constant([[1., 2., 3.]])

    with tf.GradientTape() as tape:
        # Forward pass
        y = layer(x)
        loss = tf.reduce_mean(y ** 2)

    # Calculate gradients with respect to every trainable variable
    grad = tape.gradient(loss, layer.trainable_variables)
    '''


    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['30'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['10'], help='number of network iterations')
    parser.add_argument('--acc', metavar='int', nargs=1, default=['12'], help='accelerate rate')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--learnedSVT', metavar='bool', nargs=1, default=['True'], help='Learned SVT threshold or not')

    args = parser.parse_args()
    
    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)
    
    mode = 'training'

    dataset_name = 'SIRST'
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])

    acc = int(args.acc[0])
    net_name = 'ZXY_NET'
    niter = int(args.niter[0])
    learnedSVT = bool(args.learnedSVT[0])

    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    model_id  = TIMESTAMP + net_name + '_' + dataset_name + str(acc) + '_lr_' + str(learning_rate)
    summary_writer = tf.summary.create_file_writer(os.path.join(logdir, mode, model_id + '/'))

    modeldir = os.path.join('models/stable/', model_id)
    os.makedirs(modeldir)

    # prepare dataset
    #dataset = get_dataset(batch_size, shuffle=True, full=True)
    #dataset = get_dataset('test', dataset_name, batch_size, shuffle=True, full=True)
    dataset = get_dataset_singCoil(batch_size, shuffle=True)
    tf.print('dataset loaded.')

    # initialize network
    net = ZXY_Net(niter, learnedSVT)


    tf.print('network initialized.')

    learning_rate_org = learning_rate
    learning_rate_decay = 0.95#0.95

    optimizer = tf.optimizers.Adam(learning_rate_org) #生成优化器，兼顾效率与效果
    
    
    # Iterate over epochs.
    total_step = 0
    param_num = 0
    loss = 0

    #data = dataset['DataImage']
    #label = dataset['DataLabel'] + 1
    #data = np.transpose(data, (0, 2, 1))
    #label = np.transpose(label, (0, 2, 1))
    #data = data.astype(np.float32)
    #label = label.astype(np.float32)

    for epoch in range(num_epoch):
        for step, sample in enumerate(dataset):
            
            # forward
            t0 = time.time()
            iui=0
            TrainData, TrainLabel = sample

            TrainData = tf.Variable(TrainData)
            TrainLabel = tf.Variable(TrainLabel)
            label_abs = tf.abs(TrainLabel)
            csm = None
            with tf.GradientTape() as tape:
                T, T_SYM, B = net(TrainData, TrainLabel)
                # loss = tf.reduce_mean((TrainData - T - B) ** 2/((TrainData)**2+0.0001))
                # loss = loss_function_ISTA(T, TrainLabel, T_SYM, niter)
                # MD1 = tf.reduce_mean(tf.multiply(tf.square(T - TrainLabel), TrainLabel))
                # FA1 = tf.reduce_mean(tf.multiply(tf.square(T - TrainLabel), 1 - TrainLabel))
                # loss1 = 100 * MD1 + FA1
                loss = mse(T, TrainLabel)
                # loss = loss1+loss2
            # backward
            grads = tape.gradient(loss, net.trainable_variables)####################################
            optimizer.apply_gradients(zip(grads, net.trainable_weights))#################################
            T_abs = tf.abs(T)
            # record loss
            with summary_writer.as_default():
                tf.summary.scalar('loss/total', loss.numpy(), step=total_step)

            # record gif
            '''
            if step % 20 == 0:
                with summary_writer.as_default():
                    combine_video = tf.concat([label_abs[:, :], T_abs[:, :]], axis=0).numpy()
                    combine_video = np.expand_dims(combine_video, -1)
                    video_summary('result', combine_video, step=total_step, fps=10)
            '''
            # calculate parameter number
            if total_step == 0:
                param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])

            # log output
            tf.print('Epoch', epoch+1, '/', num_epoch, 'Step', step, 'loss =', loss.numpy(), 'time', time.time() - t0, 'lr = ', learning_rate, 'param_num', param_num)
            total_step += 1

        # learning rate decay for each epoch
        learning_rate = learning_rate_org * learning_rate_decay ** (epoch + 1)#(total_step / decay_steps)
        optimizer = tf.optimizers.Adam(learning_rate)

        # save model each epoch
        #if epoch in [0, num_epoch-1, num_epoch]:
        model_epoch_dir = os.path.join(modeldir,'epoch-'+str(epoch+1), 'ckpt')
        net.save_weights(model_epoch_dir, save_format='tf')

