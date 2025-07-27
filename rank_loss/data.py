import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import TruncatedSVD

padding_length = 500


class XMLDataset(Dataset):
    def __init__(self, file, transform=None, svd=False):
        lines = [line.rstrip('\n') for line in open(file)]
        self.vocab_size, self.class_dim = [
            int(x) for x in lines[0].split()[1:]
        ]
        lines_train = lines[1:]
        self.labels = [[int(x) for x in string.split()[0].split(',')]
                       for string in lines_train]
        self.labels = np.array(self.labels, dtype=object)
        self.x_idx = [[int(x.split(':')[0]) + 1 for x in string.split()[1:]]
                      for string in lines_train]
        self.x = [
            np.array([float(x.split(':')[1])
                      for x in string.split()[1:]]).astype(np.float32)
            for string in lines_train
        ]
        has_empty = False
        for i in range(len(self.x_idx)):
            if len(self.x_idx[i]) == 0:
                self.x_idx[i] = [self.vocab_size + 1]
                self.x[i] = np.array([1.]).astype(np.float32)
                if not has_empty:
                    has_empty = True
        if has_empty:
            self.vocab_size += 1

        direct_feature = False
        try:
            self.x_idx = np.array(self.x_idx)
            self.x = np.array(self.x)
            direct_feature=True
        except:
            self.x_idx = np.array(self.x_idx, dtype=object)
            self.x = np.array(self.x, dtype=object)

        self.svd = svd

        if self.svd == True:
            self.data_matrix = np.zeros((self.x_idx.shape[0], np.max([np.max(self.x_idx[i]) for i in range(self.x_idx.shape[0])]) + 1))
            if direct_feature:
                self.data_matrix = self.x
            else:
                for i in range(self.x_idx.shape[0]):
                    self.data_matrix[i, self.x_idx[i]] = self.x[i]
            self.svd = TruncatedSVD(n_components=8, random_state=0)
            self.data_matrix = self.svd.fit_transform(self.data_matrix)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.svd == True:
            sample = ((self.data_matrix[idx], self.data_matrix[idx]), self.labels[idx])
        else:
            sample = ((self.x_idx[idx], self.x[idx]), self.labels[idx])
            if self.transform:
                sample = self.transform(sample)
        return sample


def const_padding(array, padding_length, value=0):
    if len(array) < padding_length:
        return np.pad(
            array, (0, padding_length - len(array)),
            mode='constant',
            constant_values=(0, value))
    else:
        return np.array(array[:padding_length])


def collate_fn(batch):
    x_idx = torch.stack(
        [
            torch.from_numpy(const_padding(sample[0][0], padding_length))
            for sample in batch
        ],
        dim=0)
    x = torch.stack(
        [
            torch.from_numpy(const_padding(sample[0][1], padding_length))
            for sample in batch
        ],
        dim=0)
    if isinstance(batch[0][1], list):
        label = [sample[1] for sample in batch]
    else:
        label = torch.from_numpy(np.array([sample[1] for sample in batch]))
    return [[x_idx, x], label]

def collate_fn_svd(batch):
    # print(batch[0][0][0])
    x_idx = torch.stack(
        [
            torch.from_numpy(sample[0][0]).float()
            for sample in batch
        ],
        dim=0)
    x = torch.stack(
        [
            torch.from_numpy(sample[0][1]).float()
            for sample in batch
        ],
        dim=0)
    if isinstance(batch[0][1], list):
        label = [sample[1] for sample in batch]
    else:
        label = torch.from_numpy(np.array([sample[1] for sample in batch]))
    return [[x_idx, x], label]
