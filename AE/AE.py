import os
import numpy as np
import theano
import theano.tensor as T

from theano_ops import activations
from theano_ops import optimizers
from theano_ops import utils


class AE_MODEL(object):

    def __init__(self,
                 model_object,
                 init_params=None,
                 lmbd=1e-6,
                 lr=1e-3,
                 BATCH_SIZE=64):
        self.X = T.tensor4('input')
        self.params = []
        self.to_reg = []
        self.lmbd = lmbd
        self.lr = lr
        self.BATCH_SIZE = BATCH_SIZE
        model = model_object
        encoded, params, regs = model._encode(self.X, init_params)
        self.params += params
        self.to_reg += regs
        self.X_rec, params, regs = model._decode(
            activations.sigmoid(encoded), init_params)
        self.params += params
        self.to_reg += regs
        l2_norm = np.array([T.sum(i) ** 2 for i in self.to_reg]).sum()
        cost = T.mean(
            T.sqrt(T.sum(T.sqr(self.X - self.X_rec), axis=0))) + self.lmbd * l2_norm
        self._train_op = theano.function([self.X],
                                         cost,
                                         updates=optimizers.Adam(lr=self.lr).updates(cost, self.params))
        self._reconstruct = theano.function([self.X], self.X_rec)
        self._get_code = theano.function([self.X], encoded)

    def train(self, x_train, nb_epochs):
        batch_engine = utils.BatchFactory(self.BATCH_SIZE,
                                          len(x_train),
                                          nb_epochs)
        errs = []
        iteration = 0
        nb_batches = np.ceil(1. * len(x_train) / self.BATCH_SIZE)
        batcher = batch_engine.generate_batch(x_train)
        for idx, batch in enumerate(batcher):
            errs.append(self._train_op(batch))
            if (idx + 1) % nb_batches == 0:
                iteration += 1
                print('iteration: {}\t\t train loss: {}'.format(iteration,
                                                                np.array(errs).mean()))
                print(30 * '-')
                errs = []

        if not os.path.exists('intermediate_params'):
            os.mkdir('intermediate_params')
        if not os.path.exists('intermediate_outputs'):
            os.mkdir('intermediate_outputs')
        np.save('intermediate_params/AE_params.npy', self.params)
        latent = self._get_hidden_rep(x_train)
        np.save('intermediate_outputs/latents.npy', latent)

    def _get_hidden_rep(self, X):
        batch_engine = utils.BatchFactory(
            self.BATCH_SIZE, len(X), 1, randomizer=False)
        batcher = batch_engine.generate_batch(X)
        res = []
        for idx, B in enumerate(batcher):
            res.append(self._get_code(B))
        return res

    def _reconstruct(self, X):
        batch_engine = utils.BatchFactory(
            self.BATCH_SIZE, len(X), 1, randomizer=False)
        batcher = batch_engine.generate_batch(X)
        res = []
        for idx, B in enumerate(batcher):
            res += [self._reconstruct(B)]
        return np.array(res)


if __name__ == "__main__":
    # for COIL
    # x_train = np.load('datasets/fashion-mnist/npy/fashion_x_train.npy')
    # print x_train.shape
    # print x_train.max()
    # x_train_ = x_train / 255.
    # x_train_ = x_train_.reshape((-1, 28, 28))

    # for mnist
    # import gzip
    # import pickle
    # f = gzip.open('datasets/mnist.pkl.gz', 'rb')
    # (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = pickle.load(f)
    # f.close()
    # x_train = np.load('datasets/mnist/x_train.npy')
    # y_train = np.load('datasets/mnist/y_train.npy')
    # x_test = np.load('datasets/mnist/x_test.npy')
    # y_test = np.load('datasets/mnist/y_test.npy')
    # print x_train.shape
    # print x_train.max()
    # x_train = x_train.reshape((-1, 1, 28, 28))
    # x_train_ = x_train / 255.
    # print x_train.shape, x_train.max()

    # params = np.load(
    #     'model_params/mnist_best_complete_training/conv_pars_CAE_init_joint.npy')
    # print params
    # model = AE((28, 28))
    # model.train(x_train_.astype('float32')[:10000], 500, 64)
    # np.save('AE_params_mnist_coil_like.npy', model.params)
    # R = model.reconstruct(x_train.astype('float32'))
    # print R.shape
    # np.save('mnist_rec_after_train', R)
    # x_train *= 255.
    # params1 = np.load('AE_params_coil_l1.npy')
    # params2 = np.load('AE_params_coil_l2.npy')
    # params3 = np.load('AE_params_coil_l3.npy')
    # params4 = np.load('AE_params_coil_l4.npy')
    # params = np.load('AE_params_coil.npy')
    # print params
    # x_train = np.load('datasets/stl10_binary/npy/stl_x_train.npy')
    # x_train_reshaped = x_train.swapaxes(3, 2).swapaxes(2, 1)

    # x_train_reshaped = (1. * x_train_reshaped) / 255.
    # params = np.append(params1, params2)
    # params = np.append(params, params3)
    # params = np.append(params, params4)

    # for cifar10
    # x_train = np.load('datasets/cifar10/npy/x_train.npy')
    # x_train_ = x_train / 255.
    # print x_train_.shape
    # params = np.load('AE_params_cifar10.npy')
    # model = AE((3, 32, 32), params)
    # model.train(x_train_.astype('float32'), 500, 64)
    # R = model.reconstruct(x_train_reshaped.astype('float32'))
    # np.save('stl_rec_train.npy', R)
    # encoded = model.get_hidden_rep(
    #     x_train_.astype('float32'))
    # print encoded.shape
    # np.save('AE_params_cifar10.npy', model.params)
    # np.save('AE_params_stl.npy', model.params)
    # x_corrected = []
    # for i in encoded:
    #     for j in i:
    #         x_corrected.append(j.reshape((1, -1)))
    # np.save('encoded_cifar10_train', np.array(x_corrected).squeeze())
    # np.save('encoded_stl_train_corrected.npy', np.array(x_corrected).squeeze())
    # np.save('AE_encoded_stl_train.npy', encoded)

    # for usps
    # x_train = np.load('datasets/usps/x_train_test.npy')
    # params = np.load('AE_params_usps.npy')
    # print x_train.shape
    # model = AE((16, 16), params)
    # model.train(x_train.astype('float32')[:7291], 500, 64)
    # encoded = model.get_hidden_rep(x_train.astype('float32'))
    # x_corrected = []
    # for i in encoded:
    #     for j in i:
    #         x_corrected.append(j.reshape((1, -1)))
    # np.save('encoded_usps_train_test.npy', np.array(x_corrected).squeeze())
    # np.save('AE_params_usps.npy', model.params)

    # for spine_study
    # x_train = np.load(
    #     '/home/mehran/git/auto_register/train_data/levels_dnm_input_min_black_enhanced.npy')
    # print x_train.shape
    # print x_train.max()
    # for idx in range(len(x_train)):
    #     x_train[idx] = x_train[idx] / x_train[idx].max()
    # x_train = x_train.reshape((-1, 1, 60, 90))
    # print x_train.max()
    # params = np.load('AE_params_spine_min_black_enhanced.npy')
    # model = AE((60, 90), params)
    # encoded = model.get_hidden_rep(x_train.astype('float32'))
    # x_corrected = []
    # for i in encoded:
    #     for j in i:
    #         x_corrected.append(j.reshape((1, -1)))

    # np.save('spine_min_black_enhanced_encoded.npy', x_corrected)
    # _x_train = x_train[:450].astype('float32')
    # model.train(_x_train, 500, 64)
    # np.save('AE_params_spine_min_black_enhanced.npy', model.params)

    # for yelp
    x_train = np.load(
        '/home/mehran/git/Yelp/data/train_food.npy')
    print x_train.shape
    print x_train.max()
    for idx in range(len(x_train)):
        x_train[idx] = x_train[idx] / x_train[idx].max()
    # x_train = np.swapaxes(x_train,
    print x_train.max()
    params = np.load('AE_params_spine_min_black_enhanced.npy')
    model = AE((60, 90), params)
    encoded = model.get_hidden_rep(x_train.astype('float32'))
    x_corrected = []
    for i in encoded:
        for j in i:
            x_corrected.append(j.reshape((1, -1)))

    np.save('spine_min_black_enhanced_encoded.npy', x_corrected)
