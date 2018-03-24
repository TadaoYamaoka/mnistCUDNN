﻿import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import datasets, iterators, optimizers, serializers

import argparse

from nn import NN

# 引数の定義
parser = argparse.ArgumentParser(description='example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))

# モデルの作成
model = NN()
# モデルをGPUに転送
if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

# 最適化手法の設定
optimizer = optimizers.SGD()
optimizer.setup(model)

# 保存したモデルを読み込み
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
# 保存した最適化状態を復元
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

# MNISTデータセットを読み込み
train, test = datasets.get_mnist()

train_iter = iterators.SerialIterator(train, args.batchsize)
test_iter = iterators.SerialIterator(test, args.batchsize, shuffle=False)

# 学習ループ
for epoch in range(1, args.epoch + 1):
    # ミニバッチ単位で学習
    sum_loss = 0
    itr = 0
    for i in range(0, len(train), args.batchsize):
        # ミニバッチデータ
        train_batch = train_iter.next()
        x, t = chainer.dataset.concat_examples(train_batch, args.gpu)

        # 順伝播
        y = model(x)

        # 勾配を初期化
        model.cleargrads()
        # 損失計算
        loss = F.softmax_cross_entropy(y, t)
        # 誤差逆伝播
        loss.backward()
        optimizer.update()

        sum_loss += loss.data
        itr += 1

    # 評価
    sum_test_loss = 0
    sum_test_accuracy = 0
    test_itr = 0
    for i in range(0, len(test), args.batchsize):
        # ミニバッチデータ
        test_batch = test_iter.next()
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x_test, t_test = chainer.dataset.concat_examples(test_batch, args.gpu)

                # 順伝播
                y_test = model(x_test)
                # 損失計算
                sum_test_loss += F.softmax_cross_entropy(y_test, t_test).data
                # 一致率計算
                sum_test_accuracy += F.accuracy(y_test, t_test).data
                test_itr += 1

    print('epoch={}, train loss={}, test loss={}, accuracy={}'.format(
        optimizer.epoch + 1, sum_loss / itr,
        sum_test_loss / test_itr, sum_test_accuracy / test_itr))

    optimizer.new_epoch()

# モデル保存
print('save the model')
serializers.save_npz('model', model)
# 最適化状態保存
print('save the optimizer')
serializers.save_npz('state', optimizer)
