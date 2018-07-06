import theano.tensor as T
from theano_ops import Ops, activations


def get_params(layer_name, par_list):
    if par_list is None:
        return None
    pars = []
    for i in par_list:
        if i.name.split('_')[-1] == layer_name:
            pars.append(i)
    return None if len(pars) == 0 else pars


class SIMPLE_AE(object):

    def __init__(self, input_size, latent_size):
        self.input_size = input_size
        self.latent_size = latent_size

    def _encode(self, X, init_params):
        params = []
        regs = []
        e1, pars = Ops.conv_2d(X, (10, 1, 5, 5),
                               layer_name='e1',
                               mode='half',
                               init_params=get_params('e1', init_params))
        params += pars
        regs.append(pars[0])
        e2, pars = Ops.conv_2d(activations.relu(e1),
                               (8, 10, 5, 5),
                               layer_name='e2',
                               mode='half',
                               init_params=get_params('e2', init_params))
        params += pars
        regs.append(pars[0])
        e3, pars = Ops.conv_2d(activations.relu(e2),
                               (5, 8, 5, 5),
                               layer_name='e3',
                               mode='half',
                               init_params=get_params('e3', init_params))
        params += pars
        regs.append(pars[0])
        e4, pars = Ops.dense(Ops.flatten(activations.relu(e3)),
                             self.input_size[0] * self.input_size[1] * 5,
                             self.latent_size,
                             layer_name='e4',
                             init_params=get_params('e4', init_params))
        params += pars
        regs.append(pars[0])
        return e4, params, regs

    def _decode(self, X, init_params):
        params = []
        regs = []
        d4, pars = Ops.dense(
            X, self.latent_size, self.input_size[0] * self.input_size[1] * 5,
            layer_name='d4',
            init_params=get_params('d4', init_params))
        params += pars
        regs.append(pars[0])
        d4 = T.reshape((d4),
                       (-1, 5, self.input_size[0], self.input_size[1]))

        d3, pars = Ops.conv_2d(activations.relu(d4),
                               (8, 5, 5, 5),
                               layer_name='d3',
                               mode='half',
                               init_params=get_params('d3', init_params))
        params += pars
        regs.append(pars[0])
        d2, pars = Ops.conv_2d(activations.relu(d3),
                               (10, 8, 5, 5),
                               layer_name='d2',
                               mode='half',
                               init_params=get_params('d2', init_params))
        params += pars
        regs.append(pars[0])
        d1, pars = Ops.conv_2d(activations.relu(d2),
                               (1, 10, 5, 5),
                               layer_name='d1',
                               mode='half',
                               init_params=get_params('d1', init_params))
        params += pars
        regs.append(pars[0])
        return activations.sigmoid(d1), params, regs
