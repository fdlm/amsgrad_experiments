import theano
import theano.tensor as tt
import numpy as np
from updates import adam, amsgrad
from tqdm import tqdm
from docopt import docopt


USAGE = """
Usage:
  train_synth.py [--stochastic] [--n_updates=<I>] [--plot] [--save=<S>]
                 [--lr_decay]
  
Options:
  --lr_decay  decay the learn rate according to alpha / sqrt(t)
  --stochastic  use stochastic setup
  --n_updates=<I>  number of updates to perform [default: 9000000]
  --plot  plot results after training
  --save=<S>  save results after training to this file
"""


def run_training(updater, stochastic, lr_decay, n_updates):
    x = theano.shared(np.float32(0.0))
    t = theano.shared(np.int(1))
    if stochastic:
        rs = np.random.RandomState(4711)
        rng = tt.shared_randomstreams.RandomStreams(rs.randint(999999))
        r = tt.cast(rng.binomial(n=1, p=0.01), 'floatX')
        loss = (r * 1010 - (1 - r) * 10) * x
    else:
        loss = tt.switch(tt.eq((t % 101), 1), 1010 * x, -10 * x)

    if lr_decay:
        lr = tt.cast(0.001 / tt.cast(tt.sqrt(t), 'floatX'), 'floatX')
    else:
        lr = 0.001

    updates = updater(loss, [x], learning_rate=lr,
                      beta1=0.9, beta2=0.99, bias_correction=False)
    updates[x] = tt.clip(updates[x], -1.0, 1.0)

    if not stochastic:
        updates[t] = t + 1

    train = theano.function([], loss, updates=updates)

    losses = []
    x_vals = []

    for _ in tqdm(range(n_updates)):
        losses.append(train())
        x_vals.append(x.get_value())

    return np.array(x_vals)


if __name__ == "__main__":
    args = docopt(USAGE)
    n_updates = int(args['--n_updates'])
    x_adam = run_training(adam, args['--stochastic'], args['--lr_decay'], n_updates)
    x_amsgrad = run_training(amsgrad, args['--stochastic'], args['--lr_decay'], n_updates)

    if args['--save']:
        np.savez(args['--save'], x_adam=x_adam, x_amsgrad=x_amsgrad)
    if args['--plot']:
        import matplotlib.pyplot as plt
        plt.plot(x_adam, label='adam')
        plt.plot(x_amsgrad, label='amsgrad')
        plt.legend()
        plt.show(block=True)

