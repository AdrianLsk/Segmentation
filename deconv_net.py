"""
From the paper Learning Deconvolution Network for Semantic Segmentation
http://cvlab.postech.ac.kr/research/deconvnet/
"""

import time
import os

import numpy as np
import h5py
import cPickle

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers.conv import Conv2DLayer as conv, Deconv2DLayer as deconv
from lasagne.layers.pool import MaxPool2DLayer as pool
from lasagne.layers.special import NonlinearityLayer, InverseLayer as unpool
from lasagne.layers.normalization import BatchNormLayer, batch_norm
from lasagne.nonlinearities import rectify, sigmoid
from lasagne.objectives import binary_crossentropy as bxe

from collections import OrderedDict

import theano
import theano.tensor as T
import theano.misc.pkl_utils
np.random.seed(123)

import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-lr','--learning_rate', type=float, default=0.001)
arg_parser.add_argument('-dlr', '--decrease_lr', type=float, default=1.)
arg_parser.add_argument('-bs', '--batch_size', type=int, default=6)
arg_parser.add_argument('-ep','--max_epochs', type=int, default=100)
args = arg_parser.parse_args()

def build_conv_part(net, conv_nfilts, conv_fsizes, conv_pads, conv_psizes,
                    conv_strides):
    # build convolution encoder    
    conv_settings = \
        zip(conv_nfilts, conv_fsizes, conv_pads, conv_psizes, conv_strides)
    for i, (nfilts, fsizes, pads, psize, stride) in enumerate(conv_settings):
        for j, (nfilt, fsize, pad) in enumerate(zip(nfilts, fsizes, pads)):
            net['bn_conv_{}_{}'.format(i, j)] = batch_norm(
                conv(net.values()[-1], num_filters=nfilt, pad=pad,
                     filter_size=fsize, nonlinearity=rectify))
            print net.values()[-1].output_shape
        
        if psize is not None and stride is not None:
            net['pool_{}'.format(i)] = \
                pool(net.values()[-1], pool_size=psize, stride=stride)
    
    L = i
    return net, L

def build_deconv_part(net, L, deconv_nfilts, deconv_fsizes, deconv_pads,
                      deconv_psizes):
    l = L
    # build deconvolution decoder
    for i, (nfilts, fsizes, pads, psize) in enumerate(
            zip(deconv_nfilts, deconv_fsizes, deconv_pads, deconv_psizes)
    ):
        l -= 1
        for j, (nfilt, fsize, pad) in enumerate(zip(nfilts, fsizes, pads)):
            net['bn_deconv_{}_{}'.format(l, j)] = batch_norm(
                deconv(net.values()[-1], num_filters=nfilt, crop=pad,
                       filter_size=fsize, nonlinearity=rectify))
            
            print net.values()[-1].output_shape
    
        if psize is not None:
            net['unpool_{}'.format(l)] = \
                unpool(net.values()[-1], net.get('pool_{}'.format(l)))
    
    return net

def build_model(num_classes, input_shape, conv_settings, deconv_settings):
    net = OrderedDict()
    net['input'] = InputLayer(input_shape)
    
    # build conv/pool encoder
    net, L = build_conv_part(net, *conv_settings)
    # build deconv/unpool decoder
    net = build_deconv_part(net, L, *deconv_settings)
    net['seg_score'] = conv(net.values()[-1], num_filters=num_classes,
                            filter_size=1, nonlinearity=sigmoid)

    return net['seg_score'], net


LEARNING_RATE = args.learning_rate
LR_DECREASE = args.decrease_lr
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.max_epochs

print "Loading data..."
# insert code for loading data
# imgs and seg masks should be of shape (batch_size, 1, height, width)

print "Building a model ..."
conv_nfilts = [[32]*2, [64]*2, [128]*3, [256]*3, [512]*3, [4096]*2]
conv_fsizes = [[3]*2, [3]*2, [3]*3, [3]*3, [3]*3, [7, 1]]
conv_pads = [[1]*2, [1]*2, [1]*3, [1]*3, [1]*3, [0]*2]
conv_psizes = [2, 2, 2, 2, 2, None]
conv_strides = [2, 2, 2, 2, 2, None]
conv_settings = [conv_nfilts, conv_fsizes, conv_pads, conv_psizes, conv_strides]

deconv_nfilts = [[512], [512]*2+[256], [256]*2+[128], [128]*2+[64], [64, 32],
                 [32]*2]
deconv_fsizes = [[7], [3]*3, [3]*3, [3]*3, [3]*2, [3]*2]
deconv_pads = [[0], [1]*3, [1]*3, [1]*3, [1]*2, [1]*2]
deconv_psizes = [14, 28, 56, 112, 224, None]
deconv_settings = [deconv_nfilts, deconv_fsizes, deconv_pads, deconv_psizes]

output_layer, deconv_net = build_model(2, (BATCH_SIZE, 1, 250, 250),
                                       conv_settings, deconv_settings)

# set up input/output variables
X = T.ftensor4('x')
y = T.ftensor4('y')

# training output
output_train = lasagne.layers.get_output(output_layer, X,
                                             deterministic=False)

# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(output_layer, X,
                                            deterministic=True)

net_params = lasagne.layers.get_all_params(output_layer, trainable=True) # all

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
cost = bxe(T.clip(output_train, 1e-15, 1), y).mean()

updates = lasagne.updates.adam(cost, net_params, learning_rate=sh_lr)

# get training and evaluation functions
pred = T.gt(output_eval, 0.5)
accuracy = T.eq(pred, y).mean()
ax = (1,2,3)
dice_nom = 2 * T.sum(pred * y, axis=ax) # vector
dice_denom = T.cast(pred.sum(axis=ax) + y.sum(axis=ax), 'float32') # vector
dice = T.switch(T.eq(dice_denom, 0.), 1., dice_nom / dice_denom).mean()

network_dump = {'output_layer': output_layer,
                'net': deconv_net,
                'x': X,
                'y': y,
                'output_eval': output_eval
                }

print "Compiling functions ..."
train = theano.function([X, y], [cost], updates=updates)
eval = theano.function([X, y], [cost, accuracy, dice])


def save_dump(param_values, filename):
    f = file(filename, 'wb')
    cPickle.dump(param_values,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def train_epoch():
    costs = []
    for b in range(num_batches_train):
        batch_slice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
        train_cost = \
            train(train_imgs[batch_slice] - mean, train_masks[batch_slice])

        costs.append(train_cost)

    return np.mean(costs)

def eval_epoch():
    costs = []
    accs = []
    dices = []
    for b in range(num_batches_valid):
        batch_slice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
        eval_cost, eval_acc, eval_dice = \
            eval(val_imgs[batch_slice] - mean, val_masks[batch_slice])
        costs.append(eval_cost)
        accs.append(eval_acc)
        dices.append(eval_dice)

    return np.mean(eval_cost), np.mean(eval_acc), np.mean(dices)

num_batches_train = len(train_imgs) // BATCH_SIZE
num_batches_valid = len(val_imgs) // BATCH_SIZE

experiment_name = 'deconvnet_nerve_seg_lr_{lr}_dlr_{dlr}_bs_{bs}_ep_{ep}'.format(
    lr=LEARNING_RATE, dlr=LR_DECREASE, bs=BATCH_SIZE, ep=NUM_EPOCHS
)

train_costs, valid_costs, accs, dices = [], [], [], []

print "Starting training..."
now = time.time()

try:
    for n in range(adj_epochs, NUM_EPOCHS):
        train_cost = train_epoch()
        eval_cost, acc, dice = eval_epoch()
        
        train_costs.append(train_cost)
        valid_costs.append(eval_cost)
        accs.append(acc)
        dices.append(dice)

        print "Epoch %d took %.3f s" % (n + 1, time.time() - now)
        now = time.time()
        print "Train cost {}, val cost {}, val acc {}, val dice {} " \
                .format(train_costs[-1], valid_costs[-1], accs[-1], dices[-1])
        
        if (n+1) % 10 == 0:
            new_lr = sh_lr.get_value() * LR_DECREASE
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
            # uncomment if to save the whole network
        save_dump(network_dump, 'epoch_{}_{}.pkl'.format(n, experiment_name))
        
except KeyboardInterrupt:
    pass


# uncomment if to save the learning curve
# save_dump('final_epoch_stats_{}.pkl'.format(experiment_name),
#           zip(train_costs, valid_costs, accs, dices))

# uncomment if to save the params only
# save_dump('final_epoch_params_{}.pkl'.format(experiment_name),
#           lasagne.layers.get_all_param_values(output_layer))

# uncomment if to save the whole network
save_dump(network_dump, 'final_epoch_{}.pkl'.format(experiment_name))
