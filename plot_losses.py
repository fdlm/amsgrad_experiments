import matplotlib.pyplot as plt
import yaml
from os.path import join, basename
from docopt import docopt
from itertools import izip

USAGE = """
Usage:
  plot_losses.py <exp_dirs>...
"""


def main():
    args = docopt(USAGE)
    fig = plt.figure()
    ax_loss = fig.add_subplot(121)
    ax_acc = fig.add_subplot(122)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for exp_dir, color in izip(args['<exp_dirs>'], colors):
        log = yaml.load(open(join(exp_dir, 'train_log.yaml')))
        ax_loss.plot(log['loss'], c=color, alpha=0.3)
        ax_loss.plot(log['val_loss'], c=color, alpha=1.0,
                     label=basename(exp_dir))
        ax_loss.axhline(min(log['val_loss']), c=color, lw=1, ls='--')
        ax_acc.plot(log['acc'], c=color, alpha=0.3)
        ax_acc.plot(log['val_acc'], c=color, alpha=1.0)
        ax_acc.axhline(max(log['val_acc']), c=color, lw=1, ls='--')

    ax_loss.set_title('Loss')
    ax_acc.set_title('Accuracy')
    ax_loss.legend()
    ax_acc.legend()
    plt.show(block=True)

if __name__ == "__main__":
    main()
