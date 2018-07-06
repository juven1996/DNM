import os
import theano
import theano.tensor as T
import numpy as np
from theano_ops.activations import sigmoid
from theano_ops.optimizers import Adam
from theano_ops.utils import BatchFactory


class DNM(object):

    def __init__(self,
                 input_image_size,
                 latent_size,
                 lattice_size,
                 ae_arch_class,
                 init_params=None,
                 sigma=None,
                 alpha=None,
                 BATCH_SIZE=64,
                 ae_lr=1e-3,
                 lmbd=1e-6,
                 som_pretrain_lr=-0.0005,
                 dnm_map_lr=-0.05):

        self.lattice_size = lattice_size
        self.latent_size = latent_size
        self.sigma = sigma if sigma is not None else max(self.lattice_size[0],
                                                         self.lattice_size[1]) / 2.
        self.alpha = alpha if alpha is not None else 0.3
        self.input_image_size = input_image_size
        self.ae_lr = ae_lr
        self.lmbd = lmbd
        self.ae_arch_obj = ae_arch_class(self.input_image_size,
                                         self.latent_size)
        self.BATCH_SIZE = BATCH_SIZE
        self.som_lr = som_pretrain_lr
        self.dnm_map_lr = dnm_map_lr
        self.ae_weights = None
        if init_params is not None:
            self.ae_weights = init_params[0]
        if init_params is not None:
            self.som_weights = init_params[1]
        else:
            self.som_weights = theano.shared(np.random.normal(0, 1,
                                                              (self.lattice_size[0] *
                                                               self.lattice_size[
                                                                   1],
                                                               self.latent_size)))
        if not os.path.exists('ckpts'):
            os.mkdir('ckpts')
        self._build()

    def __soft_probs(self, sample, clusters):
        alpha = 1.0
        q = 1.0 / \
            (1.0 + (T.sum(T.square((sample - clusters)), axis=1) / alpha))
        q **= (alpha + 1.0) / 2.0
        q = T.transpose(T.transpose(q) / T.sum(q, axis=0))
        return q

    def __kld(self, p, q):
        return T.sum(p * T.log(p / q), axis=-1)

    def __SOM(self, X, W, n):

        learning_rate_op = T.exp(self.som_lr * n)
        _alpha_op = self.alpha * learning_rate_op
        _sigma_op = self.sigma * learning_rate_op

        locations = self.locs
        maps = T.sub(X, W)
        measure = T.sum(T.pow(T.sub(X, W), 2), axis=1)
        err = measure.min()
        self.bmu_index = T.argmin(measure)
        bmu_loc = locations[self.bmu_index]
        dist_square = T.sum(
            T.square(T.sub(locations, bmu_loc)), axis=1)
        H = T.cast(T.exp(-dist_square / (2 * T.square(_sigma_op))),
                   dtype=theano.config.floatX)
        w_update = W + _alpha_op * \
            T.tile(H, [self.latent_size, 1]).T * maps
        Qs = self.__soft_probs(X, W)
        P = Qs ** 2 / Qs.sum()
        P = (P.T / P.sum()).T
        cost = self.__kld(P, Qs)
        return [err, cost, bmu_loc], {W: w_update}

    def _build_locs(self, m, n):
        for i in range(m):
            for j in range(n):
                yield(np.array([i, j]))

    def _build(self):
        # Tensors
        n = T.scalar()
        self.X = T.TensorType(
            dtype=theano.config.floatX, broadcastable=(False,) * 4)('X')

        # Architecture of the AE
        encoder_output, encoder_params, enc_to_regs = self.ae_arch_obj._encode(
            self.X, self.ae_weights)
        encoder_output_sigmoid = sigmoid(encoder_output)
        reconstructed, decoder_params, dec_to_regs = self.ae_arch_obj._decode(
            encoder_output_sigmoid, self.ae_weights)

        # Architecture of the SOM and updates
        self.locs = T.constant(
            [i for i in self._build_locs(self.lattice_size[0], self.lattice_size[1])])
        [self.__SOM_batch_errs, som_kl_cost, bmus], self.__SOM_updates = theano.scan(
            sequences=encoder_output_sigmoid,
            non_sequences=[self.som_weights, n],
            fn=self.__SOM)

        # Loss of DNM, SOM and AE
        ae_optimizer = Adam(lr=self.ae_lr)
        self.ae_weights = encoder_params + decoder_params
        to_regs = enc_to_regs + dec_to_regs
        self.map_err = T.mean(som_kl_cost)
        self.rec_cost = T.mean(
            T.sum(T.nnet.binary_crossentropy(reconstructed, self.X), axis=1))
        self.ae_cost = self.rec_cost + self.lmbd * \
            np.array([T.sum(i) ** 2 for i in to_regs]).sum()

        self.dnm_cost = self.ae_cost + self.map_err

        dnm_updates = ae_optimizer.updates(cost=self.dnm_cost,
                                           params=self.ae_weights)

        ae_updates = ae_optimizer.updates(cost=self.ae_cost,
                                          params=self.ae_weights)
        # Train ops
        self.__train_op_dnm = theano.function([self.X, n],
                                              outputs=[self.dnm_cost, T.mean(
                                                  self.__SOM_batch_errs)],
                                              updates=dnm_updates)
        self.__train_op_ae = theano.function([self.X],
                                             outputs=self.ae_cost,
                                             updates=ae_updates)
        self.__train_op_som = theano.function([self.X, n],
                                              outputs=self.__SOM_batch_errs,
                                              updates=self.__SOM_updates)

        self.__get_ae_output = theano.function([self.X],
                                               encoder_output_sigmoid)
        self.__get_reconstructed = theano.function([self.X],
                                                   reconstructed)
        self.__decode_embedding = theano.function([encoder_output_sigmoid],
                                                  reconstructed)
        self.__get_backprojection = theano.function([],
                                                    outputs=reconstructed,
                                                    givens={encoder_output_sigmoid: self.som_weights.eval().astype('float32')})

    def get_bmu_array(self, X):
        res = []
        weight_array = self.som_weights.get_value()
        for i in X:
            measure = np.sum((i - weight_array) ** 2, axis=1)
            bmu_index = np.argmin(measure)
            res.append(weight_array[bmu_index])
        return np.array(res)

    def train(self, x_train,
              dnm_epochs,
              pre_train_epochs=None):

        if pre_train_epochs is not None:
            self._pretrain(x_train, pre_train_epochs[
                           0], pre_train_epochs[1], self.BATCH_SIZE)
            np.save('ckpts/DNM_intermediate.npy',
                    [self.ae_weights, self.som_weights])
        self.som_lr = self.dnm_map_lr
        print('\nTraining DNM')
        print(50 * '=')
        B = BatchFactory(self.BATCH_SIZE, len(x_train), dnm_epochs)
        batcher = B.generate_batch(x_train)
        iteration = 0
        nb_batches = np.ceil(1. * len(x_train) / self.BATCH_SIZE)
        errs = []
        Ds = []
        for idx, batch in enumerate(batcher):
            self.__train_op_som(batch, iteration)
            er, dists = self.__train_op_dnm(batch, iteration)
            errs.append(er)
            Ds.append(dists)
            if (idx + 1) % (nb_batches) == 0:
                print(
                    'iter: {} \tloss: {}, dist: {}'.format(iteration + 1, np.array(errs).mean(),
                                                           np.array(Ds).mean()))
                print(50 * '-')
                iteration += 1
                Ds = []
                errs = []
        np.save('ckpts/DNM.npy',
                [self.ae_weights, self.som_weights])

    def get_locations(self, input_vects):
        to_return = []
        weight_values = list(np.array(self.som_weights.eval()))
        locations = list(np.array(self.locs.eval()))
        for vect in input_vects:
            v = vect.reshape(1, 1,
                             self.input_image_size[0],
                             self.input_image_size[1])
            min_index = min([i for i in range(len(weight_values))],
                            key=lambda x: np.linalg.norm(self.__get_ae_output(v).reshape((1, -1)) -
                                                         weight_values[x]))
            to_return.append(locations[min_index])
        return to_return

    def get_embeddings(self, X):
        res = []
        for i in X:
            res.append(self.__get_ae_output(
                i.reshape((1, 1, self.input_image_size[0], self.input_image_size[1]))))
        return np.array(res)

    def get_reconstructed(self, X):
        return self.__get_reconstructed(X)

    def decode_embedding(self, z):
        return self.__decode_embedding(z)

    def backproject_map(self):
        return self.__get_backprojection()

    def _pretrain(self, x_train, AE_epochs, SOM_epochs, BATCH_SIZE=64):
        print('\nPre-training AE')
        print(50 * '=')
        B = BatchFactory(BATCH_SIZE, len(x_train), AE_epochs)
        batcher = B.generate_batch(x_train)
        iteration = 0
        nb_batches = np.ceil(1. * len(x_train) / BATCH_SIZE)
        errs = []
        for idx, batch in enumerate(batcher):
            er = self.__train_op_ae(batch)
            errs.append(er)
            if (idx + 1) % (nb_batches) == 0:
                print('iter: {} \ttrain loss: {}'.format(iteration + 1,
                                                         np.array(errs).mean()))
                print(50 * '-')
                iteration += 1
                errs = []
                if iteration % 11 == 0:
                    np.save('ckpts/DNM_checkpoint.npy',
                            [self.ae_weights, self.som_weights])
        print('\nPre-training SOM')
        print(50 * '=')
        B = BatchFactory(BATCH_SIZE, len(x_train), SOM_epochs)
        batcher = B.generate_batch(x_train)
        iteration = 0
        nb_batches = np.ceil(1. * len(x_train) / BATCH_SIZE)
        errs = []
        for idx, batch in enumerate(batcher):
            errs.append(np.mean(self.__train_op_som(batch, iteration)))
            if (idx + 1) % nb_batches == 0:
                print('iter: {} \ttrain loss: {}'.format(iteration + 1,
                                                         np.array(errs).mean()))
                print(50 * '-')
                iteration += 1
                errs = []
                if iteration % 11 == 0:
                    np.save('ckpts/DNM_checkpoint.npy',
                            [self.ae_weights, self.som_weights])
