from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *


def vgg(in_shape, n_classes):
    """ Compile net architecture """
    nonlin = rectify
    init_conv = HeNormal

    # --- input layers ---
    l_in = InputLayer(shape=(None, in_shape[0], in_shape[1], in_shape[2]),
                      name='Input')

    # --- conv layers ---
    net = Conv2DLayer(l_in, num_filters=64, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=64, filter_size=3, pad=1, W=init_conv(),
                      nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = Conv2DLayer(net, num_filters=128, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = Conv2DLayer(net, num_filters=256, filter_size=3, pad=1,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = MaxPool2DLayer(net, pool_size=2, name='Pool')
    net = DropoutLayer(net, p=0.25, name='Dropout')

    net = Conv2DLayer(net, num_filters=1024, filter_size=3, pad=0,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5, name='Dropout')
    net = Conv2DLayer(net, num_filters=1024, filter_size=1, pad=0,
                      W=init_conv(), nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = DropoutLayer(net, p=0.5, name='Dropout')

    # --- feed forward part ---
    net = Conv2DLayer(net, num_filters=n_classes, filter_size=1, W=init_conv(),
                      nonlinearity=nonlin, name='Conv')
    net = batch_norm(net)
    net = GlobalPoolLayer(net)
    net = DenseLayer(net, num_units=n_classes, nonlinearity=softmax)
    return net
