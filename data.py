import numpy as np


def cifar10():
    xs_train = []
    ys_train = []
    for j in range(5):
        d = np.load('data/cifar-10-batches-py/data_batch_' + str(j+1))
        xs_train.append(d['data'])
        ys_train.append(d['labels'])

    X_train = np.concatenate(xs_train).reshape(-1, 3, 32, 32) / 255.
    y_train = np.concatenate(ys_train)
    pixel_mean = X_train.mean(axis=0)

    d = np.load('data/cifar-10-batches-py/test_batch')
    X_test = np.array(d['data']).reshape(-1, 3, 32, 32) / 255.
    y_test = np.array(d['labels'])

    # subtract per-pixel mean
    X_train -= pixel_mean
    X_test -= pixel_mean

    return (X_train.astype(np.float32), y_train.astype(np.int32),
            X_test.astype(np.float32), y_test.astype(np.int32))


