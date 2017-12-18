from __future__ import print_function
import numpy as np
import trattoria
import theano
import theano.tensor
from functools import partial
from docopt import docopt
import lasagne as lnn
import os

import yaml

import models
import data
import updates


USAGE = """
Usage:
  train.py --model=<S> --data=<S> --updater=<S> --n_epochs=<I> 
           --learning_rate=<F> --beta2=<F> [--bias_correction] [--run_id=<S>]
           [--no_annealing] [--l2=<F>]
"""


class Args(object):

    def __init__(self):
        args = docopt(USAGE)
        self.model = args['--model']
        self.data = args['--data']
        self.updater = args['--updater']
        self.n_epochs = int(args['--n_epochs'])
        self.learning_rate = float(args['--learning_rate'])
        self.beta2 = float(args['--beta2'])
        self.bias_correction = args['--bias_correction']
        self.run_id = args['--run_id']
        self.no_annealing = args['--no_annealing']
        self.l2 = float(args['--l2']) if args['--l2'] else None

    def dump(self, stream):
        yaml.dump(dict(
            model=self.model,
            data=self.data,
            updater=self.updater,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            beta2=self.beta2,
            bias_correction=self.bias_correction,
            run_id=self.run_id,
            no_annealing=self.no_annealing,
            l2=self.l2), stream)


def mirror_images(batches):
    for images, labels in batches:
        fl = np.random.randint(0, 2, images.shape[0])
        for i in xrange(images.shape[0]):
            if fl[i]:
                images[i] = images[i, :, :, ::-1]
        yield images, labels


def main():
    args = Args()

    create_model = getattr(models, args.model)
    load_data = getattr(data, args.data)
    updater = getattr(updates, args.updater)

    param_str = '(n={}_lr={}_b2={}_bc={})'.format(
        args.n_epochs, args.learning_rate, args.beta2, args.bias_correction)
    exp_id = '_'.join([args.model, args.data, args.updater, param_str])
    if args.run_id:
        exp_id += '_' + args.run_id
    out_dir = os.path.join('experiments', exp_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.dump(open(os.path.join(out_dir, 'config.yaml'), 'w'))

    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data()
    training_set = trattoria.data.DataSource([X_train, y_train])
    test_set = trattoria.data.DataSource([X_test, y_test])

    net = trattoria.nets.NeuralNetwork(
        create_model(X_train.shape[1:], len(np.unique(y_train))),
        theano.tensor.ivector('y')
    )

    train_batches = trattoria.iterators.AugmentedIterator(
        trattoria.iterators.BatchIterator(
            datasource=training_set,
            batch_size=128,
            shuffle=True
        ),
        mirror_images
    )
    val_batches = trattoria.iterators.BatchIterator(
        datasource=test_set,
        batch_size=1024,
        fill_last=False
    )
    val = trattoria.training.Validator(
        net=net, batches=val_batches,
        observables={
            'loss': trattoria.objectives.average_categorical_crossentropy,
            'acc': trattoria.objectives.average_categorical_accuracy
        }
    )

    lr = theano.shared(np.float32(args.learning_rate), name='learning_rate',
                       allow_downcast=True)

    if args.no_annealing:
        callbacks = []
    else:
        callbacks = [trattoria.schedules.Linear(
            lr, start_epoch=0, end_epoch=args.n_epochs, target_value=0.)]

    updater = partial(updater, learning_rate=lr, beta2=args.beta2,
                      bias_correction=args.bias_correction)

    print('\nTraining:\n')

    if args.l2:
        regularizers = [lnn.regularization.regularize_network_params(
            net.net, lnn.regularization.l2) * np.float32(args.l2)]
    else:
        regularizers = None

    trattoria.training.train(
        net=net,
        train_batches=train_batches,
        num_epochs=args.n_epochs,
        observables={
            'loss': trattoria.objectives.average_categorical_crossentropy,
            'acc': trattoria.objectives.average_categorical_accuracy,
            'lr': lambda *_: lr
        },
        updater=updater,
        validator=val,
        regularizers=regularizers,
        logs=[trattoria.outputs.ConsoleLog(),
              trattoria.outputs.YamlLog(
                  os.path.join(out_dir, 'train_log.yaml'))],
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
