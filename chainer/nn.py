from chainer import Chain
import chainer.functions as F
import chainer.links as L

# ネットワーク定義
k = 16
fcl = 256
class NN(Chain):
    def __init__(self):
        super(NN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels = 1, out_channels = k, ksize = 3, pad = 1)
            self.conv2 = L.Convolution2D(in_channels = k, out_channels = k, ksize = 3, pad = 1)
            self.l3    = L.Linear(7*7*k, fcl)
            self.l4    = L.Linear(fcl, 10)
            #self.bn1   = L.BatchNormalization(k)
            #self.bn2   = L.BatchNormalization(k)

    def __call__(self, x):
        h = self.conv1(F.reshape(x, (len(x), 1, 28, 28)))
        #h = self.bn1(h)
        h = F.max_pooling_2d(F.relu(h), 2)
        h = self.conv2(h)
        #h = self.bn2(h)
        h = F.max_pooling_2d(F.relu(h), 2)
        h = F.relu(self.l3(h))
        return self.l4(h)