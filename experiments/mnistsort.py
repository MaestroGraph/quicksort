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
import logging, time, gc
from dqsort import util
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
TAU = 16.0

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)


def gen(b, data, labels, size, digits):
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
    inds = random.choices(range(n), k=total)

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

# def plotn(data, ax):
#
#     n = data.size(0)
#
#     for i in range(n):
#         im = data[i].data.cpu().numpy()
#         ax.imshow(im, extent=(n-i-1, n-i, 0, 1), cmap='gray_r')
#
#     ax.set_xlim(0, n)
#     ax.set_ylim(0, 1)
#
#     ax.axhline()

def go(arg):
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
    if arg.final:
        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)
        trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)

        test = torchvision.datasets.MNIST(root=arg.data, train=False, download=True, transform=trans)
        testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=False, num_workers=2)

    else:
        NUM_TRAIN = 45000
        NUM_VAL = 5000
        total = NUM_TRAIN + NUM_VAL

        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)

        trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
        testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

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

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./mnistsort/{}'.format( r))

        if arg.sort_method == 'quicksort':
            model = dqsort.SortLayer(arg.size, additional=arg.additional, sigma_scale=arg.sigma_scale,
                               sigma_floor=arg.min_sigma, certainty=arg.certainty)
        else:
            model = dqsort.NeuralSort(tau=TAU)

        if arg.model == 'original':
            # architecture taken from neuralsort paper
            c1, c2, h = 32, 64, 64

            fin = (arg.digits * 28 // 4) * (28 // 4) * c2

            tokeys = nn.Sequential(
                util.Lambda(lambda x : x.view(-1, 1, 28, arg.digits * 28)),
                nn.Conv2d(1, c1, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 5, padding=2), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(c2),
                util.Flatten(),
                nn.Linear(fin, 64), nn.ReLU(),
                nn.Linear(64, 1),
                util.Lambda(lambda x : x.view(arg.batch, -1))
            )
        elif arg.model == 'big':
            # - channel sizes
            c1, c2, c3 = 16, 128, 512
            h1, h2, out = 256, 128, 8

            fin = (arg.digits * 28 // 8) * (28 // 8) * c3

            tokeys = nn.Sequential(
                util.Lambda(lambda x: x.view(-1, 1, 28, arg.digits * 28)),
                nn.Conv2d(1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c1),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c1, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c2),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c2, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1, bias=False), nn.ReLU(),
                nn.BatchNorm2d(c3),
                nn.MaxPool2d((2, 2)),
                util.Flatten(),
                nn.Linear(fin, out), nn.ReLU(),
                nn.Linear(out, 1),
                util.Lambda(lambda x : x.view(arg.batch, -1))
            )

        else:
            raise Exception('Model {} not recognized.'.format(arg.model))

        if arg.cuda:
            model.cuda()
            tokeys.cuda()

        optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

        for i in trange(arg.iterations):

            x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x)

            x = x.view(arg.batch, arg.size, -1)
            t = t.view(arg.batch, arg.size, -1)

            if type(model) == dqsort.SortLayer:
                ys, ts, keys = model(x, keys=keys, target=t)
            else:
                ys, phat = model(x, keys)

            if  arg.sort_method == 'neuralsort' and arg.loss == 'plain':
                loss = util.xent(ys, t).mean()

            elif arg.sort_method == 'neuralsort' and arg.loss == 'xent':
                _, gold = torch.sort(l, dim=1)
                loss = F.cross_entropy(phat, gold)

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

                    xb = ys[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)
                    tb = ts[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)

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

            tbw.add_scalar('mnistsort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch)

            # Plot intermediate results, and targets
            if i % arg.plot_every == 0 and arg.sort_method == 'quicksort':

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x)

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                ys, ts, _ = model(x, keys=keys, target=t)

                b, n, s = ys[0].size()

                for d in range(1, len(ys) - 1):
                    numbuckets = 2 ** d
                    bucketsize = arg.size // numbuckets

                    xb = ys[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, s)
                    tb = ts[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, s)

                    xb = xb.mean(dim=2, keepdim=True)\
                        .expand(arg.batch, numbuckets, bucketsize, s)\
                        .contiguous().view(arg.batch, n, s)
                    tb = tb.mean(dim=2, keepdim=True)\
                        .expand(arg.batch, numbuckets, bucketsize, s)\
                        .contiguous().view(arg.batch, n, s)

                    ys[d] = xb
                    ts[d] = tb

                md = int(np.log2(arg.size))
                plt.figure(figsize=(arg.size*2, md+1))

                c = 1
                for row in range(md + 1):
                    for col in range(arg.size*2):
                        ax = plt.subplot(md+1, arg.size*2, c)

                        images = ys[row] if col < arg.size else ts[row]
                        im = images[0].view(arg.size, 28, arg.digits * 28)[col%arg.size].data.cpu().numpy()

                        ax.imshow(im, cmap='gray_r')
                        clean(ax)

                        c += 1

                plt.savefig('./mnistsort/{}/intermediates.{:04}.pdf'.format(r, i))

            # Plot the progress
            if i % arg.plot_every == 0 and arg.sort_method == 'quicksort':

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x)
                keys.retain_grad()

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                yt, _ = model(x, keys=keys, train=True)

                loss = F.mse_loss(yt, t)  # compute the loss
                loss.backward()

                yi, _ = model(x, keys=keys, train=False)

                input  = x[0].view(arg.size, 28, arg.digits*28)
                target = t[0].view(arg.size, 28, arg.digits*28)
                output_inf   = yi[0].view(arg.size, 28, arg.digits*28)
                output_train = yt[0].view(arg.size, 28, arg.digits*28)

                plt.figure(figsize=(arg.size*3*arg.digits, 4*3))
                for col in range(arg.size):

                    ax = plt.subplot(4, arg.size, col + 1)
                    ax.imshow(target[col].data.cpu().numpy())
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('target')

                    ax = plt.subplot(4, arg.size, col + arg.size + 1)
                    ax.imshow(input[col].data.cpu().numpy())
                    clean(ax)
                    ax.set_xlabel( '{:.2}, {:.2}'.format(keys[0, col], - keys.grad[0, col] ) )

                    if col == 0:
                        ax.set_ylabel('input')

                    ax = plt.subplot(4, arg.size, col + arg.size * 2 + 1)
                    ax.imshow(output_inf[col].data.cpu().numpy())
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('inference')

                    ax = plt.subplot(4, arg.size, col + arg.size * 3 + 1)
                    ax.imshow(output_train[col].data.cpu().numpy())
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('training')

                plt.savefig('./mnistsort/{}/mnist.{:04}.pdf'.format(r, i))

            if i % arg.dot_every == 0:
                """
                Compute the accuracy
                """
                print('Finished iteration {}, repeat {}/{}, computing accuracy'.format(i, r, arg.reps))
                NUM = 10_000
                tot, tot_sub = 0.0, 0.0
                correct, sub = 0.0, 0.0

                with torch.no_grad():

                    for test_size in arg.test_sizes:
                        for _ in range(NUM//arg.batch):
                            x, t, l = gen(arg.batch, data_test, labels_test, test_size, arg.digits)

                            if arg.cuda:
                                x, _, l = x.cuda(), t.cuda(), l.cuda()

                            x, l = Variable(x), Variable(l)

                            keys = tokeys(x)

                            # Sort the keys, and sort the labels, and see if the resulting indices match
                            _, gold = torch.sort(l, dim=1)
                            _, mine = torch.sort(keys, dim=1)

                            tot += x.size(0)
                            correct += ((gold != mine).sum(dim=1) == 0).sum().item()

                            sub += (gold == mine).sum()
                            tot_sub += util.prod(gold.size())

                        print('test size {}, accuracy {:.5} ({:.5})'.format( test_size, correct/tot, float(sub)/tot_sub) )

                    tbw.add_scalar('mnistsort/testloss/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch)


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Whether to use the original model, or a bigger version.",
                        default='original', type=str)

    parser.add_argument("--sort",
                        dest="sort_method",
                        help="Whether to use the baseline (NeuralSort), or Quicksort.",
                        default='quicksort', type=str)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Dimensionality of the input.",
                        default=128, type=int)

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

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Number of iterations (in batches).",
                        default=8000, type=int)

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
                        help="How many iterations per dot in the loss curves.",
                        default=1000, type=int)

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

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=1, type=int)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    parser.add_argument("-L", "--limit",
                        dest="limit",
                        help="Limit on the nr ofexamples per class (for debugging).",
                        default=None, type=int)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set.",
                        action="store_true")

    parser.add_argument("-I", "--loss",
                        dest="loss",
                        help="Whether to backwards-sort the target to provide a loss at every step.",
                        default="separate", type=str)


    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
