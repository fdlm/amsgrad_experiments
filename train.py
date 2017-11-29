from __future__ import print_function
import numpy as np
import trattoria
import theano
import theano.tensor
from functools import partial
import sys
import os

import yaml

import models
import data
import updates


def mirror_images(batches):
    for images, labels in batches:
        fl = np.random.randint(0, 2, images.shape[0])
        for i in xrange(images.shape[0]):
            if fl[i]:
                images[i] = images[i, :, :, ::-1]
        yield images, labels


def main():
    if len(sys.argv) < 4:
        print('Usage:\n  train.py <model> <data> <updater> [<updater_params>]')
        return 1

    create_model = getattr(models, sys.argv[1])
    load_data = getattr(data, sys.argv[2])
    updater = getattr(updates, sys.argv[3])

    if len(sys.argv) == 5:
        print(sys.argv[4])
        updater_params = yaml.load(sys.argv[4])
        print(updater_params)
    else:
        updater_params = {}

    param_str = yaml.dump(updater_params).replace(' ', '_').strip()
    out_dir = os.path.join(
        'experiments',
        sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + param_str)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    yaml.dump(dict(model=sys.argv[1], data=sys.argv[2], updater=sys.argv[3],
                   updater_params=updater_params),
              open(os.path.join(out_dir, 'config.yaml'), 'w'))

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

    updater = partial(updater, **updater_params)

    print('\nTraining:\n')

    trattoria.training.train(
        net=net,
        train_batches=train_batches,
        num_epochs=500,
        observables={
            'loss': trattoria.objectives.average_categorical_crossentropy,
            'acc': trattoria.objectives.average_categorical_accuracy,
        },
        updater=updater,
        validator=val,
        logs=[trattoria.outputs.ConsoleLog(),
              trattoria.outputs.YamlLog(
                  os.path.join(out_dir, 'train_log.yaml'))],
    )


if __name__ == "__main__":
    main()
