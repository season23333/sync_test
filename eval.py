import torch
import argparse
from random_classes import random_classes
from sync_test import sync_test
import pickle
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', default=128)
# parse.add_argument('--lr', default=0.0001)

args = parse.parse_args()


def eval(dataset):
    corpus = dataset.batchify('tst', args.batch_size)
    n = len(corpus)

    model = sync_test()

    if torch.cuda.is_available():
        model.cuda()

    model.load_state_dict(torch.load('model.pkl'))

    # optimizer = torch.optim.Adam(model.parameters(), args.lr)

    model.eval()

    val_loss = 0
    step = 0
    for i in range(n):
        ource, wizard, target = corpus[i]

        out, hieeden_state = model(corpus[i])

        predicts = torch.argmax(out.detach(), dim=1)
        corrects = torch.eq(predicts, target.cuda().long()).float().sum()
        total = len(target)

        loss = torch.nn.functional.cross_entropy(out, target.cuda().long())
        acc = corrects.detach().unsqueeze(0) / total * 100

        val_loss += loss.item()
        step += 1

    print('Test Loss: %.3f, Perplexity: %5.2f, acc: %.3f' % (val_loss / step, np.exp(val_loss / step), acc))


if __name__ == '__main__':
    dataset = torch.load('dataset.pt')
    eval(dataset)
