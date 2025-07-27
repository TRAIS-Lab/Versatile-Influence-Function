import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693,
                       torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >=
        10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y**2 / 24 - y**4 /
        2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        log1mexp(y)  # Hope for the best
    )


def nll_topk_one_sample(w, A, c=5., T=10000, clamp_min=None):
    """ Calculates the negative log-likelihood of the first partition items.
    w: (1, N).
    A: (1, K).
    """
    if clamp_min:
        w = w.clamp(min=clamp_min)

    A_onehot = torch.zeros_like(w)  # (1, N)
    A_onehot[0, A[0]] = 1
    A_onehot = A_onehot.to(dtype=torch.bool)
    S_onehot = torch.bitwise_not(A_onehot)
    w_S = torch.logsumexp(w[S_onehot].view(1, -1) + c, dim=-1)  # (1,)
    w_a_set = w[A_onehot].view(1, -1)  # (1, K)

    # log_v = (torch.arange(100, T + 100, out=w.new()) / (T + 100)).log()
    log_v = (torch.arange(100, T + 100, device=w.device, dtype=w.dtype) / (T + 100)).log()

    _q = gumbel_log_survival(
        -((w_a_set + c)[None, :, :] + torch.log(-log_v)[:, None, None]))
    q = _q.sum(-1) + (torch.expm1(w_S)[None, :] * log_v[:, None])
    sum_q = torch.logsumexp(q, 0)

    return -sum_q - w_S


class CELoss(object):
    def __init__(self, seed=0, average=False, clamp_min=-10):
        self.rs = np.random.RandomState(seed)
        self.ce = nn.CrossEntropyLoss()
        self.average = average
        self.clamp_min = clamp_min

    def __call__(self, outputs, labels):
        pred = outputs
        # if self.clamp_min:
        #     pred = pred.clamp(min=self.clamp_min)

        if not self.average:
            single_labels = torch.from_numpy(
                np.array([self.rs.choice(y) for y in labels])).to(outputs.device)
            return self.ce(pred, single_labels)

        loss = []
        log_probs = F.log_softmax(outputs, dim=-1)
        for w, A in zip(log_probs, labels):
            A = torch.from_numpy(np.array(A)).to(
                dtype=torch.long, device=w.device)
            loss.append(- w[A].mean())
        return torch.stack(loss).mean()


class TopKLoss(object):
    def __init__(self, c=5., T=10000, clamp_min=-10):
        self.c = c
        self.T = T
        self.clamp_min = clamp_min

    def __call__(self, outputs, labels):
        loss = []
        for w, A in zip(outputs, labels):
            A = torch.from_numpy(np.array(A)).to(w.device)
            w = w.view(1, -1)
            A = A.view(1, -1)
            loss.append(
                nll_topk_one_sample(
                    w, A, self.c, self.T, self.clamp_min))
        return torch.stack(loss).mean()


def surrogate(z, func_type="hinge"):
    if func_type == "hinge":
        return F.relu(1 - z).sum()
    elif func_type == "logistic":
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
        logits = z.view(-1, 1)
        return criterion(logits, torch.ones_like(logits))
    else:
        raise NotImplementedError(
            "Surrogate type {} not supported.".format(func_type))


def rank_loss_one_sample(w, A, func_type="hinge", max_neg_samples=40000):
    temp = w.detach()
    A_onehot = torch.zeros_like(temp)  # (1, N)
    A_onehot[0, A[0]] = 1
    A_onehot = A_onehot.to(dtype=torch.bool)
    S_onehot = torch.bitwise_not(A_onehot)
    if S_onehot.size(1) > max_neg_samples:
        random_idx = torch.randperm(S_onehot.size(1))[max_neg_samples:]
        random_idx = random_idx.to(device=S_onehot.device).view(1, -1)
        S_onehot.scatter_(1, random_idx, False)
    pairs_loss = surrogate(w[A_onehot].view(1, -1) - w[S_onehot].view(-1, 1),
                           func_type)
    return pairs_loss


class RankLoss(object):
    def __init__(self, seed=0, func_type="hinge", max_neg_samples=40000):
        self.func_type = func_type
        self.max_neg_samples = max_neg_samples

    def __call__(self, outputs, labels):
        loss = []
        for w, A in zip(outputs, labels):
            A = torch.from_numpy(np.array(A)).to(w.device)
            w = w.view(1, -1)
            A = A.view(1, -1)
            loss.append(
                rank_loss_one_sample(
                    w, A, func_type=self.func_type,
                    max_neg_samples=self.max_neg_samples))
        return torch.stack(loss).mean()
