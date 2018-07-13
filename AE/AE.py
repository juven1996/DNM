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
