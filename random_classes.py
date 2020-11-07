import os, sys, time, random, re, json, spacy, torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from spacy.tokenizer import Tokenizer
# from enchant.checker import SpellChecker

# from .. import datasets_utils
# from .. import register_dataset
# from . import basic_label_dataset

# @register_dataset('random_classes')
class random_classes():
    def __init__(self):
        # self.task_type = 'test'
        self.num_label = 10
        # dataset_config.num_label = self.num_label
        # self.config = dataset_config
        trn_num = 2000
        val_num = 100
        tst_num = 100
        
        embed_dim = 300
        centers = (np.random.rand(self.num_label, embed_dim) - 0.5) * 0.5  # 10*300
        stds = np.random.rand(embed_dim)  # 300

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(trn_num):
                target.append(np.array([label]))  # label
                source.append(np.random.normal(center, std, [1, embed_dim]))  # source 300
            label += 1

        self.source_trn = np.concatenate(source, axis = 0).astype(np.float32)  # 20000*300
        self.wizard_trn = None
        self.target_trn = np.concatenate(target)

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(val_num):
                target.append(np.array([label]))
                source.append(np.random.normal(center, std, [1, embed_dim]))
            label += 1

        self.source_val = np.concatenate(source, axis = 0).astype(np.float32)
        self.wizard_val = None
        self.target_val = np.concatenate(target)

        source = []
        target = []
        label = 0
        for center, std in zip(centers, stds):
            for idx in range(tst_num):
                target.append(np.array([label]))
                source.append(np.random.normal(center, std, [1, embed_dim]))
            label += 1

        self.source_tst = np.concatenate(source, axis = 0).astype(np.float32)
        self.wizard_tst = None
        self.target_tst = np.concatenate(target)
    
    # def print_self(self):
    #     print('hahha')

    def shuffle(self, source, wizard, target):
        np.random.seed(1)
        indices = np.arange(len(target))
        np.random.shuffle(indices)
        source = source[indices]
        if wizard is not None and len(wizard) > 0:
            wizard = wizard[indices]
        else:
            wizard = None
        if target is not None and len(target) > 0:
            target = target[indices]
        else:
            target = None

        return source, wizard, target

    def batchify(self, tvt, batch_size, pad_mode = 'post', shuffle = True, same_len = True, seq_len = None):
        if tvt == 'trn':
            source_tmp = self.source_trn
            wizard_tmp = self.wizard_trn
            target_tmp = self.target_trn
        elif tvt == 'val':
            source_tmp = self.source_val
            wizard_tmp = self.wizard_val
            target_tmp = self.target_val
        elif tvt == 'tst':
            source_tmp = self.source_tst
            wizard_tmp = self.wizard_tst
            target_tmp = self.target_tst
        else:
            raise ValueError('tvt value must in [trn, val, txt].')

        if seq_len is None: # not netgram
            assert len(source_tmp) == len(target_tmp), f'The length of data and label must be same [{len(source_tmp)} and {len(target_tmp)}].'
            assert pad_mode in ['pre', 'post'], 'pad_mode must in [pre, post], got [{}]'.format(pad_mode)

            if shuffle == True:
                source_tmp, wizard_tmp, target_tmp = self.shuffle(source_tmp, wizard_tmp, target_tmp)

            nbatch = len(target_tmp) // batch_size
            if not len(target_tmp) % batch_size == 0: nbatch += 1
        # else: # netgram
        #     len(data) // (self.seq_len * batch_size)
        #     data_tmp = source_tmp
        #     source_tmp = data_tmp[:num_batch * batch_size * seq_len].reshape([-1, seq_len])
        #     target_tmp = data_tmp[1: num_batch * batch_size * seq_len + 1].reshape([-1, seq_len])
        #     same_len = False

        ret = []
        idx = 0
        for iterator in range(nbatch):
            if idx + batch_size < len(target_tmp):
                stmp = source_tmp[idx:idx + batch_size]
                wtmp = wizard_tmp[idx:idx + batch_size] if wizard_tmp is not None else None
                ttmp = target_tmp[idx:idx + batch_size]
            elif same_len:
                stmp = source_tmp[-batch_size:]
                wtmp = wizard_tmp[-batch_size:] if wizard_tmp is not None else None
                ttmp = target_tmp[-batch_size:]
            else:
                stmp = source_tmp[idx:idx + batch_size]
                wtmp = wizard_tmp[idx:idx + batch_size] if wizard_tmp is not None else None
                ttmp = target_tmp[idx:idx + batch_size]
            idx += batch_size

            stmp = torch.from_numpy(stmp)
            wtmp = torch.from_numpy(wtmp) if wtmp is not None else None
            ttmp = torch.from_numpy(ttmp)

            ret.append([stmp, wtmp, ttmp])
        return ret

    # @classmethod
    # def setup_dataset(cls):
    #     return cls






















