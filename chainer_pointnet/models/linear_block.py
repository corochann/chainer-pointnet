import chainer
from chainer import functions
from chainer import links


class LinearBlock(chainer.Chain):

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu, dropout_ratio=-1, residual=False):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.linear = links.Linear(
                in_size, out_size=out_size, nobias=nobias,
                initialW=initialW, initial_bias=initial_bias)
            if use_bn:
                self.bn = links.BatchNormalization(out_size)
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        if self.use_bn:
            h = self.bn(self.linear(x))
        else:
            h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            from chainerex.functions import residual_add
            h = residual_add(h, x)
        if self.dropout_ratio >= 0:
            h = functions.dropout(h, ratio=self.dropout_ratio)
        return h
