import os, sys
import torch
import argparse
from sync_test import sync_test
from random_classes import random_classes
import torch.optim as optim
import torch.nn as nn
import numpy as np

parse = argparse.ArgumentParser()
parse.add_argument('--lr', default=0.0001)
parse.add_argument('--epochs', default=25)
parse.add_argument('--batch_size', default=128)
parse.add_argument('--clip', default=5)

args = parse.parse_args()


def validate(model):
    val_loss = 0
    step = 0

    corpus = dataset.batchify('val', args.batch_size)
    n = len(corpus)
    for j in range(n):
        source, wizard, target = corpus[j]

        out, hieeden_state = model(corpus[j])

        predicts = torch.argmax(out.detach(), dim=1)
        corrects = torch.eq(predicts, target.cuda().long()).float().sum()
        total = len(target)

        loss = torch.nn.functional.cross_entropy(out, target.cuda().long())
        acc = corrects.detach().unsqueeze(0) / total * 100

        val_loss += loss.item()
        step += 1

        model.zero_grad()

    print('Validation Loss: %.3f, Perplexity: %5.2f, acc: %.3f' % (val_loss / step, np.exp(val_loss / step), acc))

    return val_loss / step


def train(dataset):
    cur_best = 10000

    corpus = dataset.batchify('trn', args.batch_size)
    # print(len(corpus))  # batch_num 157, source, w, target, batch_size 128, embedding_dim 300
    n = len(corpus)

    model = sync_test()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        model.train()

        for i in range(n):

            model.zero_grad()

            source, wizard, target = corpus[i]
            # print(source.size())  # 128*300

            out, hieeden_state = model(corpus[i])
            # print(out)
            predicts = torch.argmax(out.detach(), dim=1)
            corrects = torch.eq(predicts, target.cuda().long()).float().sum()
            total = len(target)

            loss = torch.nn.functional.cross_entropy(out, target.cuda().long())
            acc = corrects.detach().unsqueeze(0) / total * 100
            # print(target.size())

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                print('Epoch %d/%d, Loss: %.3f, Perplexity: %5.2f, Acc: %.3f' % (
                    epoch, args.epochs, loss.item(), np.exp(loss.item()), acc))

        model.eval()
        val_loss = validate(model)
        val_perplex = np.exp(val_loss)

        if val_perplex < cur_best:
            print("The current best val loss: ", val_loss)
            cur_best = val_perplex
            torch.save(model.state_dict(), 'model.pkl')


if __name__ == '__main__':
    dataset = random_classes()
    torch.save(dataset, 'dataset.pt')
    train(dataset)
