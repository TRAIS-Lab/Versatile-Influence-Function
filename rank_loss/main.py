import argparse
import os
import time
from copy import deepcopy
from six.moves import cPickle as pickle
from multiprocessing import Pool, cpu_count
from functools import partial

from utils import Logger

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import XMLDataset, collate_fn, collate_fn_svd
from loss import CELoss, RankLoss, TopKLoss
from models import MLP, MLP_WO_Embedding, LR_WO_Embedding
from torch.utils.data import DataLoader
from utils import (PrecisionRecall, PropensityPrecisionRecall,
                   NormalizedDCG, PropensityNormalizedDCG)
from utils import calc_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=2, help="Verbose level.")
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument(
    "--accumulation_splits", type=int, default=1,
    help="Split batch_size with multiple backwards to fit the GPU memory.")
parser.add_argument(
    "--log_interval", type=int, default=20, help="Log interval.")
parser.add_argument("--log_path", default="results", help="Path to save logs.")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--loss", default="nll_partition", help="Type of loss.")
parser.add_argument("--dataset", default="Wiki10", help="Dataset.")
parser.add_argument("--train_ratio", type=float, default=0.75,
                    help="The ratio of training data used to train.")
parser.add_argument(
    "--has_feature",
    action="store_true",
    help="Whether the dataset has real-value features.")
parser.add_argument(
    "--emb_dim", type=int, default=256, help="Embedding dimension.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument(
    "--weight_decay", type=float, default=5e-4, help="Weight decay.")
parser.add_argument(
    "--grad_clip", type=float, default=0, help="Gradient clip.")
parser.add_argument(
    "--emb_scale_grad",
    action="store_true",
    help="Scale embedding gradients by frequency.")
parser.add_argument("--valid_key", type=int, default=2, help="Valid metric.")
parser.add_argument(
    "--k_list", nargs="+", default=[1, 3, 5, 10], type=int,
    help="List of k for evaluation metrics.")
parser.add_argument(
    "--ps_valid", action="store_true", help="Whether use PS metric for valid.")
parser.add_argument("--clamp_min", type=float, default=-10,
                    help="Clamp the logits with a min value.")
parser.add_argument(
    "--ce_average",
    action="store_true",
    help="Sample one label for CE if False; average over all labels if True.")
parser.add_argument(
    "--max_neg_samples",
    type=int,
    default=40000,
    help="Max number of negative samples for pairwise losses.")
parser.add_argument(
    "--remove_label",
    type=int,
    default=-1,
    help="Remove one of the label during the training.")
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="training seed.")
parser.add_argument(
    "--svd",
    action="store_true",
    help="Use SVD to reduce the dimension of the feature.")
args = parser.parse_args()

KNN_MIN_DIM = 50000

global_start_time = time.time()

if args.log_path is not None:
    folder = os.path.join(
        args.log_path, args.dataset,
        "feat_{}__ratio_{}__ep_{}__interval_{}".format(
            args.has_feature, int(args.train_ratio * 100), args.epoch,
            args.log_interval))
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file_name = "loss_{}__bs_{}__lr_{}__dim_{}__stamp_{}".format(
        args.loss, args.batch_size, args.lr, args.emb_dim,
        str(int(global_start_time)))
    lgr = Logger(args.verbose, log_path=folder, file_prefix=log_file_name)
else:
    lgr = Logger(args.verbose)

# torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

batch_size = args.batch_size // args.accumulation_splits
emb_dim = args.emb_dim
k_list = args.k_list
valid_k_idx = args.valid_key

lgr.p(args.dataset)
lgr.p(args)
file_prefix = "./data/{}/{}{}_".format(args.dataset, args.dataset[0].lower(),
                                       args.dataset[1:])
if os.path.exists(file_prefix + "train.txt"):
    dataset_train = XMLDataset(file_prefix + "train.txt")
    dataset_test = XMLDataset(file_prefix + "test.txt")
else:
    file_name = "./data/{}/{}_data.txt".format(args.dataset, args.dataset)
    dataset = XMLDataset(file_name, svd=args.svd)
    train_idx = np.loadtxt(file_prefix + "trSplit.txt", dtype=int)[:, 0] - 1
    test_idx = np.loadtxt(file_prefix + "tstSplit.txt", dtype=int)[:, 0] - 1
    if args.svd:
        dataset_train = deepcopy(dataset)
        dataset_train.labels = dataset_train.labels[train_idx]
        dataset_train.x = dataset_train.data_matrix[train_idx]
        dataset_train.x_idx = dataset_train.data_matrix[train_idx]
        dataset_test = deepcopy(dataset)
        dataset_test.labels = dataset_test.labels[test_idx]
        dataset_test.x = dataset_test.data_matrix[test_idx]
        dataset_test.x_idx = dataset_test.data_matrix[test_idx]
    else:
        dataset_train = deepcopy(dataset)
        dataset_train.labels = dataset_train.labels[train_idx]
        dataset_train.x = dataset_train.x[train_idx]
        dataset_train.x_idx = dataset_train.x_idx[train_idx]
        dataset_test = deepcopy(dataset)
        dataset_test.labels = dataset_test.labels[test_idx]
        dataset_test.x = dataset_test.x[test_idx]
        dataset_test.x_idx = dataset_test.x_idx[test_idx]

train_size = int(args.train_ratio * len(dataset_train))
dataset_val = deepcopy(dataset_train)
dataset_train.labels = dataset_train.labels[:train_size]
dataset_train.x = dataset_train.x[:train_size]
dataset_train.x_idx = dataset_train.x_idx[:train_size]
dataset_val.labels = dataset_val.labels[train_size:]
dataset_val.x = dataset_val.x[train_size:]
dataset_val.x_idx = dataset_val.x_idx[train_size:]

# CHANGE!
if args.dataset == "Mediamill2":
    index = []
    for i, label in enumerate(dataset_train.labels):
        if len(label) > 1:
            index.append(i)
        if len(index) == 5000:
            break
    dataset_train.labels = dataset_train.labels[index]
    dataset_train.x = dataset_train.x[index]
    dataset_train.x_idx = dataset_train.x_idx[index]

    index = []
    for i, label in enumerate(dataset_val.labels):
        if len(label) > 1:
            index.append(i)
    dataset_val.labels = dataset_val.labels[index]
    dataset_val.x = dataset_val.x[index]
    dataset_val.x_idx = dataset_val.x_idx[index]

eval_batch_size = args.batch_size * 10

if args.svd:
    collate_fn = collate_fn_svd

train_loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(
    dataset_val, batch_size=eval_batch_size, shuffle=False,
    collate_fn=collate_fn)
test_loader = DataLoader(
    dataset_test, batch_size=eval_batch_size, shuffle=False,
    collate_fn=collate_fn)


def eval(loader):
    model.eval()

    val_labels = []
    val_rankings = []
    with torch.no_grad():
        for j, val_batch in enumerate(loader):
            val_feature, val_label = val_batch
            val_x_idx, val_x = val_feature
            val_x_idx, val_x = map(lambda v: v.to(args.device),
                                   [val_x_idx, val_x])
            if hasattr(model, "knn_search") and class_dim > KNN_MIN_DIM:
                if args.has_feature:
                    val_ranking = model.knn_search(
                        k=max(args.k_list), x_idx=val_x_idx, x=val_x)
                else:
                    val_ranking = model.knn_search(
                        k=max(args.k_list), x_idx=val_x_idx)
            else:
                if args.has_feature:
                    val_outputs = model.forward(val_x_idx, val_x)
                else:
                    val_outputs = model.forward(val_x_idx)
                val_ranking = torch.topk(
                    val_outputs, k=max(args.k_list), dim=1)[1].cpu().numpy()
            val_labels.append(np.array(val_label, dtype=object))
            val_rankings.append(val_ranking)

    val_labels = np.concatenate(val_labels, axis=0, dtype=object)
    val_rankings = np.concatenate(val_rankings, axis=0, dtype=object)

    metrics = [pr_metric, pspr_metric, ndcg_metric, psndcg_metric]
    with Pool(cpu_count()) as pool:
        all_metrics = pool.map(
            partial(calc_metrics, metrics=metrics, labels=val_labels,
                    rankings=val_rankings),
            list(range(val_labels.shape[0])))
    val_prec_recall = np.array([l[0] for l in all_metrics])
    val_ps_prec_recall = np.array([l[1] for l in all_metrics])
    val_ndcg = np.array([l[2] for l in all_metrics])
    val_ps_ndcg = np.array([l[3] for l in all_metrics])

    lgr.p("Precision @ %s:   " % str(k_list) + ", ".join([
        "%.4f" % prec for prec in val_prec_recall.mean(axis=0)[:len(k_list)]
    ]))
    lgr.p("Recall @ %s:      " % str(k_list) + ", ".join([
        "%.4f" % recall
        for recall in val_prec_recall.mean(axis=0)[len(k_list):]
    ]))
    lgr.p("NDCG @ %s:        " % str(k_list) + ", ".join([
        "%.4f" % ndcg for ndcg in val_ndcg.mean(axis=0)
    ]))

    lgr.p("PS-Precision @ %s:   " % str(k_list) + ", ".join([
        "%.4f" % prec for prec in val_ps_prec_recall.mean(axis=0)[:len(k_list)]
    ]))
    lgr.p("PS-Recall @ %s:      " % str(k_list) + ", ".join([
        "%.4f" % recall
        for recall in val_ps_prec_recall.mean(axis=0)[len(k_list):]
    ]))
    lgr.p("PS-NDCG @ %s:        " % str(k_list) + ", ".join([
        "%.4f" % ndcg for ndcg in val_ps_ndcg.mean(axis=0)
    ]))
    model.train()
    return val_prec_recall, val_ps_prec_recall, val_ndcg, val_ps_ndcg


if args.loss in ["nll_partition_lb", "attn_rank"]:
    criterion = CELoss(average=args.ce_average)
elif args.loss == "nll_partition":
    criterion = TopKLoss()
elif "rank" in args.loss:
    temp = args.loss.split("_")[1]
    if temp == "net":
        func_type = "logistic"
    else:
        func_type = "hinge"
    criterion = RankLoss(func_type=func_type,
                         max_neg_samples=args.max_neg_samples)
else:
    raise NotImplementedError("Loss type {} not supported!".format(args.loss))

vocab_size = train_loader.dataset.vocab_size
class_dim = train_loader.dataset.class_dim
if args.svd:
    # model = MLP_WO_Embedding(100, 8, class_dim)
    model = LR_WO_Embedding(8, class_dim)
else:
    model = MLP(vocab_size, emb_dim, class_dim,
                emb_scale_grad=args.emb_scale_grad)
model = model.to(args.device)
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# record logs if args.log_path is not an empty string
if args.log_path:
    logs = {}
    logs["args"] = args
    logs["results"] = []

A = 0.55
B = 1.5
pr_metric = PrecisionRecall(k=k_list)
pspr_metric = PropensityPrecisionRecall(labels=dataset_train.labels,
                                        num_labels=class_dim, A=A, B=B,
                                        k=k_list)
ndcg_metric = NormalizedDCG(k=k_list)
psndcg_metric = PropensityNormalizedDCG(labels=dataset_train.labels,
                                        num_labels=class_dim, A=A, B=B,
                                        k=k_list)
total_steps = 0
running_loss = []
start_time = time.time()
best_metric = 0
st = time.time()
for epoch in range(args.epoch):
    for i, sample_batched in enumerate(train_loader):
        if (len(train_loader) - i) <= (
                len(train_loader) % args.accumulation_splits):
            continue
        features, labels = sample_batched
        idx_to_remove = []
        for label_batch_idx, label in enumerate(labels):
            if len(label) == 1:
                idx_to_remove.append(True)
            else:
                idx_to_remove.append(False)
        if args.remove_label != -1:
            for label_batch_idx, label in enumerate(labels):
                if args.remove_label in label:
                    label.remove(args.remove_label)
 
        x_idx, x = features
        x_idx, x = map(lambda v: v.to(args.device), [x_idx, x])

        if sum(idx_to_remove) > 0:
            x_idx = x_idx[~torch.tensor(idx_to_remove)]
            x = x[~torch.tensor(idx_to_remove)]
            new_labels = []
            for idx, label in enumerate(labels):
                if idx_to_remove[idx] == 0:
                    new_labels.append(label)
            labels = np.array(new_labels, dtype=object)

        if args.has_feature:
            outputs = model.forward(x_idx, x)
        else:
            outputs = model.forward(x_idx)
        if args.remove_label != -1:
            outputs[:, args.remove_label] = -torch.inf
        loss = criterion(outputs, labels)  # loss
        loss = loss / args.accumulation_splits
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        running_loss.append(loss.item())

        if (i + 1) % args.accumulation_splits == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_steps += 1
        else:
            continue

        if total_steps % args.log_interval == 0:
            the_step = (i + 1) // args.accumulation_splits
            epoch_step = len(train_loader) // args.accumulation_splits
            run_time = int(time.time() - start_time)
            start_time = time.time()
            with torch.no_grad():
                train_loss = np.mean(running_loss)
                lgr.p("Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, "
                      "Time: {}:{}".format(epoch + 1, args.epoch, the_step,
                                           epoch_step, train_loss,
                                           run_time // 60, run_time % 60))
                running_loss = []

                outputs_sort = torch.argsort(outputs)
                unique_list = [
                    torch.unique(outputs_sort[:, -idx]).size(0)
                    for idx in k_list
                ]
                lgr.p("Unique @ %s:      " % str(k_list) + ", ".join(
                    ["%5d" % uniq for uniq in unique_list]))

                if hasattr(model, "build_index") and class_dim > KNN_MIN_DIM:
                    model.build_index()

print(time.time() - st)
# torch.save(model.state_dict(), f"checkpoints_lr/model_{args.dataset}_remove_label_{args.remove_label}_seed_{args.seed}_dim_8_epoch{args.epoch}.pth")
eval(val_loader)
