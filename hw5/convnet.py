import numpy as np
from layers import *
from fast_layers import *
from layer_utils import *


class ConvNet(object):

    def __init__(self, conv_dims, hidden_dims, input_dim=(3, 32, 32), num_classes=10, loss_fuction='softmax',
                 weight_scale=1e-3, reg=0.0, dtype=np.float32):

        self.num_conv_layers = len(conv_dims)
        self.num_fc_layers = len(hidden_dims)
        self.loss_fuction = loss_fuction
        self.reg = reg
        self.dtype = dtype
        self.params = {}

        # conv_layer
        for i in range(self.num_conv_layers):
            if i == 0:
                self.params['CW1'] = weight_scale * np.random.randn(
                    conv_dims[i][0], input_dim[0], conv_dims[i][1], conv_dims[i][1])
                self.params['CB1'] = np.zeros(conv_dims[i][0])

            else:
                self.params['CW' + str(i + 1)] = weight_scale * np.random.randn(
                    conv_dims[i][0], conv_dims[i - 1][0], conv_dims[i][1], conv_dims[i][1])
                self.params['CB' + str(i + 1)] = np.zeros(conv_dims[i][0])

        # affine layer
        for i in range(self.num_fc_layers):
            if i == 0:
                self.params['W' + str(i + 1)] = weight_scale * \
                    np.random.randn(
                        conv_dims[-1][0] * input_dim[1] * input_dim[2] / 4**self.num_conv_layers, hidden_dims[i])
                self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])

            # to output layer
            elif i == self.num_fc_layers - 1:
                self.params['W' + str(i + 1)] = weight_scale * \
                    np.random.randn(hidden_dims[i - 1], num_classes)
                self.params['b' + str(i + 1)] = np.zeros(num_classes)

            else:
                self.params['W' + str(i + 1)] = weight_scale * \
                    np.random.randn(hidden_dims[i - 1], hidden_dims[i])
                self.params['b' + str(i + 1)] = np.zeros(hidden_dims[i])

        self.conv_params = []
        self.conv_params = [{'stride': 1, 'pad': (
            conv_dims[i][1] - 1) / 2} for i in range(self.num_conv_layers)]
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        cache = []

        a = X
        for i in range(self.num_conv_layers):
            a_out, conv_cache = conv_relu_pool_forward(a, self.params['CW' + str(
                i + 1)], self.params['CB' + str(i + 1)], self.conv_params[i], self.pool_param)

            a = a_out
            cache.append(conv_cache)

        M = X.shape[0]
        x_temp_shape = a.shape

        for i in range(self.num_fc_layers):
            a_out, affine_cache = affine_forward(
                a.reshape(M, -1), self.params['W' + str(i + 1)], self.params['b' + str(i + 1)])

            a = a_out
            cache.append(affine_cache)

        scores = a

        if mode == 'test':
            return scores

        loss, grads = 0, {}

        if self.loss_fuction == 'svm_loss':
            loss, dscores = svm_loss(scores, y)
        else:
            loss, dscores = softmax_loss(scores, y)

        # regularization

        for i in range(self.num_conv_layers):
            loss += 0.5 * self.reg * np.sum(self.params['CW' + str(i + 1)]**2)

        for i in range(self.num_fc_layers):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i + 1)]**2)

        dout = dscores

        for i in reversed(range(self.num_fc_layers)):
            dout_cache, grads['W' + str(i + 1)], grads['b' +
                                                       str(i + 1)] = affine_backward(dout, cache.pop(-1))
            dout = dout_cache
        dout = dout.reshape(x_temp_shape)
        for i in reversed(range(self.num_conv_layers)):
            dout_cache, grads['CW' + str(i + 1)], grads['CB' + str(
                i + 1)] = conv_relu_pool_backward(dout, cache.pop(-1))
            dout = dout_cache

        return loss, grads
