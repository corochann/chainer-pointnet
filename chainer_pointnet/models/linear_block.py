import chainer
from chainer import functions
from chainer import links


class LinearBlock(chainer.Chain):

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None, use_bn=True,
                 activation=functions.relu):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.linear = links.Linear(
                in_size, out_size=out_size, nobias=nobias,
                initialW=initialW, initial_bias=initial_bias)
            if use_bn:
                self.bn = links.BatchNormalization(out_size)
        self.activation = activation
        self.use_bn = use_bn

    def __call__(self, x):
        if self.use_bn:
            h = self.bn(self.linear(x))
        else:
            h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        return h
