from _context import dqsort

import torch, random
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import logging, time, gc, sys, copy
from dqsort import util, det_neuralsort
import numpy as np

from argparse import ArgumentParser

import torchvision
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Experiment: learn to sort numbers consisting of multiple MNIST digits.

"""

tbw = SummaryWriter()

util.DEBUG = False
BUCKET_SIGMA = 0.05

DATA_SEED = 0
TRAIN_SIZE = 100_000
TEST_SIZE = 10_000

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)


def gen(b, data, labels, size, digits, inds=None):
    """

    :param b:
    :param data: An MNIST tensor containing the images (test or training)
    :param labels: the corresponding labels
    :param size:
    :param digits: number of digits per instances
    :return: x, t, l. The permuted digits, the sorted digits and the label (the represented number as an integer).
    """

    n = data.size(0)

    total = b * size * digits
    if inds is None:
        inds = random.choices(range(n), k=total)
    else:
        assert len(inds) == total, 'Index size ({}) should be batch * size * digits ({} * {} * {} = {})'.format(len(inds), b, size, digits, total)

    x   = data[inds, :, :, :]
    l = labels[inds]

    x = x.view(b, size, digits, 1, 28, 28)
    l = l.view(b, size, digits)

    power = 10 ** torch.arange(digits, dtype=torch.long)
    l = (l * power).sum(dim=2)

    _, idx = l.sort(dim=1)

    t = x.gather(dim=1, index=idx[:, :, None, None, None, None].expand(b, size, digits, 1, 28, 28))

    x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
    x = x.view(b, size, 1, 28, 28 * digits)

    t = t.permute(0, 1, 3, 4, 2, 5).contiguous()
    t = t.view(b, size, 1, 28, 28 * digits)

    return x, t, l

class Keynet(nn.Module):

    def __init__(self, batchnorm=False, small=True, num_digits=4):
        super().__init__()

        if small:
            # architecture taken from neuralsort paper
            c1, c2, h = 32, 64, 64

            fin = (num_digits * 28 // 4) * (28 // 4) * c2

            self.tokeys = nn.Sequential(
                nn.Conv2d(1, c1, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(c1) if batchnorm else util.Lambda(lambda x: x),
                nn.Conv2d(c1, c2, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(c2) if batchnorm else util.Lambda(lambda x: x),
                util.Flatten(),
                nn.Linear(fin, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        else:
            # - channel sizes
            c1, c2, c3 = 16, 128, 512
            h1, h2, out = 256, 128, 8

            fin = (num_digits * 28 // 8) * (28 // 8) * c3

            self.tokeys = nn.Sequential(
                nn.Conv2d(1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c1) if batchnorm else util.Lambda(lambda x: x),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c1, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c2) if batchnorm else util.Lambda(lambda x: x),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c2, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c3) if batchnorm else util.Lambda(lambda x: x),
                nn.MaxPool2d((2, 2)),
                util.Flatten(),
                nn.Linear(fin, out), nn.ReLU(),
                nn.Linear(out, 1)
            )

    def forward(self, x):
        b, n = x.size(0), x.size(1)

        x = x.view(b * n, 1, 28, -1)
        y = self.tokeys(x)

        return y.view(b, n)

def go(arg, verbose=True):
    """

    :param arg:
    :return:
    """

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    random.seed(arg.seed)

    torch.set_printoptions(precision=10)

    """
    Load and organize the data
    """
    trans = torchvision.transforms.ToTensor()
    if arg.split == 'final':
        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)
        trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)

        test = torchvision.datasets.MNIST(root=arg.data, train=False, download=True, transform=trans)
        testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=False, num_workers=2)

    elif arg.split == 'validation':
        NUM_TRAIN = 45000
        NUM_VAL = 5000
        total = NUM_TRAIN + NUM_VAL

        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)

        trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
        testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    elif arg.split == 'search':
        NUM_TRAIN = 40000
        NUM_VAL = 5000
        total = NUM_TRAIN + NUM_VAL

        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)

        trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
        testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))
    else:
        assert False, 'Split mode {} not recognized'.format(arg.split)

    shape = (1, 28, 28 * arg.digits)
    num_classes = 10

    # Load the MNIST digits into a single tensor
    xbatches = []
    lbatches = []
    for xbatch, lbatch in trainloader:
        xbatches.append(xbatch)
        lbatches.append(lbatch)

    data   = torch.cat(xbatches, dim=0)
    labels = torch.cat(lbatches, dim=0)

    if arg.limit is not None:
        data = data[:arg.limit]
        labels = labels[:arg.limit]

    xbatches = []
    lbatches = []
    for xbatch, lbatch in testloader:
        xbatches.append(xbatch)
        lbatches.append(lbatch)

    data_test   = torch.cat(xbatches, dim=0)
    labels_test = torch.cat(lbatches, dim=0)

    tr = lambda n : int((n*n - n) // 2)
    train_size = TRAIN_SIZE // tr(arg.size)
    # -- reduce the train size, to make the number of pair comparisons constant
    #    note that we only fix the training size to a given number. The test set is re-sampled each time, because it just
    #    improves the accuracy estimate

    if verbose:
        print('training on {} unique permutations of size {}.'.format(train_size, arg.size))

    # sample 100K permutations for the training set
    rand = random.Random(DATA_SEED)
    s = arg.digits * arg.size
    train_perms = [rand.choices(range(data.size(0)), k=s) for _ in range(train_size)]

    util.makedirs('./mnistsort/')

    if arg.sort_method == 'quicksort':
        model = dqsort.SortLayer(arg.size, additional=arg.additional, sigma_scale=arg.sigma_scale,
                           sigma_floor=arg.min_sigma, certainty=arg.certainty)
    else:
        model = dqsort.NeuralSort(tau=arg.temp)

    if arg.model == 'original':
       tokeys = Keynet(batchnorm=arg.batch_norm, small=True, num_digits=arg.digits)
    elif arg.model == 'big':
       tokeys = Keynet(batchnorm=arg.batch_norm, small=False, num_digits=arg.digits)
    else:
        raise Exception('Model {} not recognized.'.format(arg.model))

    if arg.cuda:
        model.cuda()
        tokeys.cuda()

    optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

    seen = 0
    accuracy, proportion = 0.0, 0.0

    for e in (range(arg.epochs) if verbose else trange(arg.epochs)):
        if verbose:
            print('epoch', e)

        if arg.resample:
            s = arg.digits * arg.size
            train_perms = [rand.choices(range(data.size(0)), k=s) for _ in range(train_size)]

        for fr in (trange(0, train_size, arg.batch) if verbose else range(0, train_size, arg.batch)):

            to = min(train_size, fr+arg.batch)
            ind = train_perms[fr:to]
            ind = [item for sublist in ind for item in sublist] # flatten

            x, t, l = gen(to-fr, data, labels, arg.size, arg.digits, inds=ind)

            b = x.size(0)

            if arg.cuda:
                x, t, l = x.cuda(), t.cuda(), l.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x)

            x = x.view(b, arg.size, -1)
            t = t.view(b, arg.size, -1)

            if type(model) == dqsort.SortLayer:
                ys, ts, keys = model(x, keys=keys, target=t)
            else:
                ys, phat, phatraw = model(x, keys)

            if arg.sort_method == 'neuralsort' and arg.loss == 'plain':
                loss = util.xent(ys, t).mean()

            elif arg.sort_method == 'neuralsort' and arg.loss == 'xent':

                _, gold = torch.sort(l, dim=1)
                loss = F.cross_entropy(phatraw.permute(0, 2, 1), gold)

            elif arg.loss == 'plain':
                # just compare the output to the target
                loss = util.xent(ys[-1], t).mean()

            elif arg.loss == 'means':
                # compare the output to the back-sorted target at each step
                loss = 0.0
                loss = loss + util.xent(ys[0], ts[0]).mean()
                loss = loss + util.xent(ts[-1], ts[-1]).mean()

                # average over the buckets
                for d in range(1, len(ys)-1):
                    numbuckets = 2 ** d
                    bucketsize = arg.size // numbuckets

                    xb = ys[d][:, None, :, :].view(b, numbuckets, bucketsize, -1)
                    tb = ts[d][:, None, :, :].view(b, numbuckets, bucketsize, -1)

                    xb = xb.mean(dim=2)
                    tb = tb.mean(dim=2)

                    loss = loss + util.xent(xb, tb).mean() * bucketsize

            elif arg.loss == 'separate':
                loss = 0.0

                for d in range(len(ys)):
                    loss = loss + util.xent(ys[0], ts[0]).mean()

            else:
                raise Exception('Loss {} not recognized.'.format(arg.loss))

            loss.backward()

            optimizer.step()

            seen += to - fr
            tbw.add_scalar('mnistsort/loss/{}'.format(arg.size), loss.data.item(), seen)

        if verbose and (e % arg.dot_every == 0 or e == arg.epochs - 1):
            """
            Compute the accuracy
            """
            print('computing accuracy')

            with torch.no_grad():

                for test_size in arg.test_sizes:

                    tot, tot_sub = 0.0, 0.0
                    correct, sub = 0.0, 0.0

                    for fr in range(0, TEST_SIZE, arg.batch):

                        x, t, l = gen(arg.batch, data_test, labels_test, test_size, arg.digits)

                        if arg.cuda:
                            x, _, l = x.cuda(), t.cuda(), l.cuda()

                        x, l = Variable(x), Variable(l)

                        keys = tokeys(x)

                        if arg.sort_method == 'neuralsort':
                            keys = - keys

                        # Sort the keys, and sort the labels, and see if the resulting indices match
                        _, gold = torch.sort(l, dim=1)
                        _, mine = torch.sort(keys, dim=1)

                        tot += x.size(0)
                        correct += ((gold != mine).sum(dim=1) == 0).sum().item()

                        sub += (gold == mine).sum()
                        tot_sub += util.prod(gold.size())

                    if test_size == arg.size:
                        accuracy = correct/tot
                        proportion = float(sub)/tot_sub

                    print('test size {}, accuracy {:.5} ({:.5})'.format( test_size, correct/tot, float(sub)/tot_sub) )
                    tbw.add_scalar('mnistsort/test-acc/{}'.format(test_size), correct/tot, e)

    return accuracy, proportion


opt_arg = None
opt_acc = -1.0

def sweep(arg):
    carg = copy.deepcopy(arg)

    sizes = arg.test_sizes
    carg.test_sizes = [arg.size] # only check the accuracy on the training set size
    carg.split = 'validation' if arg.split == 'final' else 'search'

    hyperparams = {'lr' : [1e-3, 1e-4, 1e-5], 'batch' : [64]}

    if arg.sort_method == 'neuralsort':
        hyperparams['temp'] = [2, 4, 8]
    else:
        hyperparams['temp'] = [-1]

    sweep_inner(carg, hyperparams)

    opt_arg.test_sizes = sizes
    opt_arg.split = 'final' if arg.split == 'final' else 'validation'

    return opt_arg

def sweep_inner(carg, hyperparams, depth = 0):
    global opt_arg, opt_acc

    if depth == len(hyperparams):
        carg = copy.deepcopy(carg)

        for k, v in hyperparams.items():
            setattr(carg, k, v)

        print('starting sweep with', hyperparams)
        acc, _ = go(carg, verbose=False)

        if acc > opt_acc:
            opt_acc = acc
            opt_arg = carg

        return

    param = list(hyperparams.keys())[depth]

    for value in hyperparams[param]:

        chp = copy.deepcopy(hyperparams)
        chp[param] = value

        sweep_inner(carg, chp, depth + 1)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Whether to use the original model, or a bigger version.",
                        default='original', type=str)

    parser.add_argument("--sort",
                        dest="sort_method",
                        help="Whether to use the baseline (neuralsort), or quicksort.",
                        default='quicksort', type=str)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Dimensionality of the input.",
                        default=16, type=int)

    parser.add_argument("--test-sizes",
                        dest="test_sizes",
                        nargs="+",
                        help="Dimensionality of the test data (default is same as the input data).",
                        default=[4, 8, 16, 32], type=int)

    parser.add_argument("-w", "--width",
                        dest="digits",
                        help="Number of digits in each number sampled.",
                        default=3, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs over the fixed set of permutations",
                        default=100, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled globally",
                        default=2, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Plot every x iterations",
                        default=50, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-d", "--dot-every",
                        dest="dot_every",
                        help="How many epochs between computing the accuracy",
                        default=5, type=int)

    parser.add_argument("-D", "--data",
                        dest="data",
                        help="Data ditectory.",
                        default='./data', type=str)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale.",
                        default=0.1, type=float)

    parser.add_argument("-C", "--certainty",
                        dest="certainty",
                        help="Certainty: scaling factor in the bucketing computation.",
                        default=10.0, type=float)

    parser.add_argument("-T", "--temperature",
                        dest="temp",
                        help="Temperature for the neuralsort baseline.",
                        default=16, type=int)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    parser.add_argument("--key-mult",
                        dest="key_mult",
                        help="multiplier for the keys (helps with the gradient)",
                        default=1.0, type=float)

    parser.add_argument("-L", "--limit",
                        dest="limit",
                        help="Limit on the nr ofexamples per class (for debugging).",
                        default=None, type=int)

    parser.add_argument("--split", dest="split",
                        help="Whether to run on the final test set, the validation set, or a subset of the training set.",
                        default='validation', type=str)

    parser.add_argument("--batch-norm",
                        dest="batch_norm",
                        help="Whether to use batch normalization in the key net.",
                        action="store_true")

    parser.add_argument("--sweep",
                        dest="sweep",
                        help="Whether to perform a parameter sweep before running the final experiment.",
                        action="store_true")

    parser.add_argument("--resample",
                        dest="resample",
                        help="Whether resample the permutations every epoch (providing unbounded examples).",
                        action="store_true")

    parser.add_argument("-I", "--loss",
                        dest="loss",
                        help="Whether to backwards-sort the target to provide a loss at every step.",
                        default="separate", type=str)

    options = parser.parse_args()

    print('Options:', options)

    if(options.sweep):
        options = sweep(options)
        print('Selected hyperparameters:', options)

    a, p = go(options)

    print(options)
    print('final acc {:.3}, prop {:.3}'.format(a, p))
