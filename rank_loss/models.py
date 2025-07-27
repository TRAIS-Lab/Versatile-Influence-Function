import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
import numpy as np

class LR_WO_Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LR_WO_Embedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x

class MLP_WO_Embedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_WO_Embedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    """MLP model."""

    def __init__(self, vocab_size, emb_dim, class_dim, emb_scale_grad=True,
                 gpu_search=True):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.class_dim = class_dim
        self.embed = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0,
                                  scale_grad_by_freq=emb_scale_grad)
        self.linear = nn.Linear(emb_dim, class_dim)

        self.y_index = None
        self.faiss_res = None
        self.gpu_search = gpu_search
        if gpu_search:
            self.faiss_res = faiss.StandardGpuResources()

    def forward(self, x_idx, x=None):
        embed = self.embed(x_idx)
        if x is not None:
            size = x.size()
            sqrt = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).view(size[0], 1, 1)
            embed = embed * (x / sqrt.view(-1, 1)).unsqueeze(2)
        else:
            sqrt = torch.sqrt(torch.sum(
                torch.ge(x_idx, 1), dim=1).to(dtype=torch.float32))
            indicators = torch.zeros_like(x_idx)
            indicators[torch.ge(x_idx, 1)] = 1
            indicators = indicators.to(dtype=embed.dtype)
            embed = embed * (indicators / sqrt.view(-1, 1)).unsqueeze(2)

        embed = torch.sum(embed, dim=1)
        # return self.linear(F.relu(embed))
        return self.linear(embed)

    def build_index(self):
        self.y_index = faiss.IndexFlatIP(self.emb_dim + 1)  # max inner product
        y_weights = self.linear.weight.data.cpu().numpy()
        y_weights = np.concatenate(
            [y_weights, self.linear.bias.data.cpu().numpy().reshape(-1, 1)],
            axis=1)
        if self.gpu_search:
            torch.cuda.empty_cache()
            self.faiss_res.setTempMemory(int(1024**3 * 2))  # 2 GB
            self.y_index = faiss.index_cpu_to_gpu(self.faiss_res, 0, self.y_index)
        self.y_index.add(y_weights)

    def del_index(self):
        self.y_index.reset()
        del self.y_index
        self.y_index = None
        self.faiss_res.noTempMemory()

    def knn_search(self, k, x_idx, x=None):
        embed = self.embed(x_idx)
        if x is not None:
            size = x.size()
            # embed *= x.view(size[0], size[1], 1)
            sqrt = torch.sqrt(torch.sum(torch.pow(x, 2), dim=-1)).view(size[0], 1, 1)
            embed = embed * (x / sqrt.view(-1, 1)).unsqueeze(2)
        else:
            sqrt = torch.sqrt(torch.sum(
                torch.ge(x_idx, 1), dim=1).to(dtype=torch.float32))
            indicators = torch.zeros_like(x_idx)
            indicators[torch.ge(x_idx, 1)] = 1
            indicators = indicators.to(dtype=embed.dtype)
            embed = embed * (indicators / sqrt.view(-1, 1)).unsqueeze(2)

        embed = F.relu(torch.sum(embed, dim=1))  # (batch_size, emb_dim)
        embed = torch.cat([embed, torch.ones_like(embed[:, 0:1])], dim=-1)

        if self.gpu_search:
            _, knn = search_index_pytorch(self.y_index, embed, k)
            knn = knn.cpu().numpy()
        else:
            _, knn = self.y_index.search(embed.cpu().numpy(), k)
        return knn


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_long_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I
