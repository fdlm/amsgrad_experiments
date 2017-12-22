import numpy as np
import os


def mirror_images(batches):
    for images, labels in batches:
        fl = np.random.randint(0, 2, images.shape[0])
        for i in xrange(images.shape[0]):
            if fl[i]:
                images[i] = images[i, :, :, ::-1]
        yield images, labels


def no_augmentation(batches):
    for images, labels in batches:
        yield images, labels


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

    return (X_train.astype(np.float32), y_train.astype(np.int32),
            X_test.astype(np.float32), y_test.astype(np.int32), mirror_images)


def mnist():
    from urllib import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    # X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    # y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    pixel_mean = X_train.mean(axis=0)
    X_train -= pixel_mean
    X_val -= pixel_mean

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, no_augmentation  #, X_test, y_test
