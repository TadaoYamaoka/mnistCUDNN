import chainer
from chainer import cuda
from chainer import datasets, iterators, serializers

import argparse

from nn import NN

parser = argparse.ArgumentParser(description='example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=2,
                    help='Number of images in each mini-batch')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-m', default='model',
                    help='Initialize the model from given file')
args = parser.parse_args()

# モデルの作成
model = NN()
# モデルをGPUに転送
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

# 保存したモデルを読み込み
print('Load model from', args.initmodel)
serializers.load_npz(args.initmodel, model)

# MNISTデータセットを読み込み
train, test = datasets.get_mnist()

test_iter = iterators.SerialIterator(test, args.batchsize, shuffle=False)

# ミニバッチデータ
test_batch = test_iter.next()
with chainer.no_backprop_mode():
    with chainer.using_config('train', False):
        x_test, t_test = chainer.dataset.concat_examples(test_batch, args.gpu)

        # 順伝播
        y_test = model(x_test)

print(y_test.data)