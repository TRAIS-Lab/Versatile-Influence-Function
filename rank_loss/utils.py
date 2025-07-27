import os

import numpy as np


class Logger(object):

    def __init__(self, verbose=0, log_path=None, file_prefix=""):
        self.verbose = verbose
        self.filename = None
        if log_path is not None:
            self.filename = os.path.join(
                log_path, file_prefix + ".log")
            with open(self.filename, "w") as f:
                f.write(self.filename)
                f.write("\n")

    def p(self, s, level=1):
        if self.verbose >= level:
            print(s)
        if self.filename is not None:
            with open(self.filename, "a") as f:
                f.write(str(s))
                f.write("\n")


class BaseMetric(object):

    def __init__(self, rel_threshold, k):
        self.rel_threshold = rel_threshold
        if np.isscalar(k):
            k = np.array([k])
        self.k = k

    def __len__(self):
        return len(self.k)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _compute(self, *args, **kwargs):
        raise NotImplementedError


class PrecisionRecall(BaseMetric):

    def __init__(self, rel_threshold=0, k=10):
        super(PrecisionRecall, self).__init__(rel_threshold, k)

    def __len__(self):
        return 2 * len(self.k)

    def __str__(self):
        str_precision = [('Precision@%1.f' % x) for x in self.k]
        str_recall = [('Recall@%1.f' % x) for x in self.k]
        return (','.join(str_precision)) + ',' + (','.join(str_recall))

    def __call__(self, targets, predictions):
        precision, recall = zip(*[
                                  self._compute(targets, predictions, x)
                                  for x in self.k
                                  ])
        result = np.concatenate((precision, recall), axis=0)
        return result

    def _compute(self, targets, predictions, k):
        predictions = predictions[:k]
        num_hit = len(set(predictions).intersection(set(targets)))

        return float(num_hit) / len(predictions), float(num_hit) / len(targets)


class PropensityPrecisionRecall(BaseMetric):

    def __init__(self, labels, num_labels=None, A=0.55, B=1.5, rel_threshold=0,
                 k=10):
        super(PropensityPrecisionRecall, self).__init__(rel_threshold, k)
        num_instances = len(labels)
        C = (np.log(num_instances) - 1) * np.power(B + 1, A)
        flat_labels = [item for sublist in labels for item in sublist]
        if not num_labels:
            num_labels = max(flat_labels) + 1
        unique, counts = np.unique(flat_labels, return_counts=True)
        freqs = np.zeros((num_labels,))
        freqs[unique] = counts
        self.ips = 1.0 + C * np.power(freqs + B, -A)

    def __len__(self):
        return 2 * len(self.k)

    def __str__(self):
        str_precision = [('PS-Precision@%1.f' % x) for x in self.k]
        str_recall = [('PS-Recall@%1.f' % x) for x in self.k]
        return (','.join(str_precision)) + ',' + (','.join(str_recall))

    def __call__(self, targets, predictions):
        precision, recall = zip(*[
                                  self._compute(targets, predictions, x)
                                  for x in self.k
                                  ])
        result = np.concatenate((precision, recall), axis=0)
        return result

    def _compute(self, targets, predictions, k):
        predictions = predictions[:k]
        hits = list(set(predictions).intersection(set(targets)))
        target_ips = self.ips[targets]
        ps_rec = float(sum(self.ips[hits])) / sum(target_ips)
        if len(target_ips) > k:
            ind = np.argpartition(target_ips, -k)[-k:]
            target_ips = target_ips[ind]
        ps_prec = float(sum(self.ips[hits])) / sum(target_ips)

        return ps_prec, ps_rec


class MeanAP(BaseMetric):

    def __init__(self, rel_threshold=0, k=np.inf):
        super(MeanAP, self).__init__(rel_threshold, k)

    def __call__(self, targets, predictions):
        result = [self._compute(targets, predictions, x)
                  for x in self.k
                  ]
        return np.array(result)

    def __str__(self):
        return ','.join([('MeanAP@%1.f' % x) for x in self.k])

    def _compute(self, targets, predictions, k):
        if len(predictions) > k:
            predictions = predictions[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predictions):
            if p in targets and p not in predictions[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not list(targets):
            return 0.0

        return score / min(len(targets), k)


class NormalizedDCG(BaseMetric):

    def __init__(self, rel_threshold=0, k=10):
        super(NormalizedDCG, self).__init__(rel_threshold, k)

    def __call__(self, targets, predictions):
        result = [self._compute(targets, predictions, x)
                  for x in self.k
                  ]
        return np.array(result)

    def __str__(self):
        return ','.join([('NDCG@%1.f' % x) for x in self.k])

    def _compute(self, targets, predictions, k):
        k = min(len(targets), k)

        if len(predictions) > k:
            predictions = predictions[:k]

        # compute idcg
        idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
        dcg = 0.0
        for i, p in enumerate(predictions):
            if p in targets:
                dcg += 1 / np.log2(i + 2)
        ndcg = dcg / idcg

        return ndcg


class PropensityNormalizedDCG(BaseMetric):

    def __init__(self, labels, num_labels=None, A=0.55, B=1.5, rel_threshold=0,
                 k=10):
        super(PropensityNormalizedDCG, self).__init__(rel_threshold, k)
        num_instances = len(labels)
        C = (np.log(num_instances) - 1) * np.power(B + 1, A)
        flat_labels = [item for sublist in labels for item in sublist]
        if not num_labels:
            num_labels = max(flat_labels) + 1
        unique, counts = np.unique(flat_labels, return_counts=True)
        freqs = np.zeros((num_labels,))
        freqs[unique] = counts
        self.ips = 1.0 + C * np.power(freqs + B, -A)

    def __call__(self, targets, predictions):
        result = [self._compute(targets, predictions, x)
                  for x in self.k
                  ]
        return np.array(result)

    def __str__(self):
        return ','.join([('PS-NDCG@%1.f' % x) for x in self.k])

    def _compute(self, targets, predictions, k):
        k = min(len(targets), k)

        if len(predictions) > k:
            predictions = predictions[:k]

        ps_dcg = 0.0
        for i, p in enumerate(predictions):
            if p in targets:
                ps_dcg += self.ips[p] / np.log2(i + 2)  # replace 1 by ips[p]

        # normalize by max possible ps-dcg
        target_ips = self.ips[targets]
        if len(target_ips) > k:
            ind = np.argpartition(target_ips, -k)[-k:]
            target_ips = target_ips[ind]
        ps_idcg = np.sum(target_ips / np.log2(np.arange(2, k + 2)))

        ps_ndcg = ps_dcg / ps_idcg

        return ps_ndcg


def calc_metrics(idx, metrics, labels, rankings):
    return [metric(labels[idx], rankings[idx]) for metric in metrics]
