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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level.")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs.")
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

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

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
    print(vocab_size, class_dim)
    if args.svd:
        # model = MLP_WO_Embedding(100, 8, class_dim)
        model = LR_WO_Embedding(8, class_dim)
    else:
        model = MLP(vocab_size, emb_dim, class_dim,
                    emb_scale_grad=args.emb_scale_grad)
    model = model.to(args.device)

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

    model.load_state_dict(torch.load(f"checkpoints_lr/model_{args.dataset}_remove_label_-1_seed_0_dim_8.pth"))
    eval(val_loader)

    from dattri.func.utils import flatten_func, flatten_params
    from torch.func import grad, hessian

    model.eval()
    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
    print(model_params.keys(), model_params["fc1.weight"].shape,
                               model_params["fc1.bias"].shape)
    # exit(0)

    with torch.no_grad():
        @flatten_func(model)
        def train_loss(params, sample_batched, remove_label=None, test=False):
            features, labels = sample_batched

            idx_to_remove = []
            for label_batch_idx, label in enumerate(labels):
                if len(label) == 1:
                    idx_to_remove.append(True)
                else:
                    idx_to_remove.append(False)
            if test is True:
                idx_to_remove = [False] * len(labels)

            # assert sum(idx_to_remove) == 0

            if remove_label is not None:
                for label in labels:
                    if remove_label in label:
                        label.remove(remove_label)
            x_idx, x = features
            x_idx, x = map(lambda v: v.to(args.device), [x_idx, x])

            if sum(idx_to_remove) == len(labels):
                return torch.tensor(0.0).to(args.device)

            if sum(idx_to_remove) > 0:
                x_idx = x_idx[~torch.tensor(idx_to_remove)]
                x = x[~torch.tensor(idx_to_remove)]
                new_labels = []
                for idx, label in enumerate(labels):
                    if idx_to_remove[idx] == 0:
                        new_labels.append(label)
                labels = np.array(new_labels, dtype=object)

            outputs = torch.func.functional_call(model, params, x_idx)  # (x_idx, x)
            if remove_label is not None:
                outputs[:, remove_label] = -torch.inf

            loss = criterion(outputs, labels)
            return loss

        @flatten_func(model)
        def output_logit(params, sample_batched):
            features, labels = sample_batched
            x_idx, x = features
            x_idx, x = map(lambda v: v.to(args.device), [x_idx, x])
            outputs = torch.func.functional_call(model, params, x_idx)
            return torch.sum(outputs[:, labels])

        full_gradient = 0
        hess = 0
        normalization_count = 0
        for i, sample_batched in enumerate(train_loader):
            len_sample_batched = len(sample_batched[1])
            full_gradient = (normalization_count/(normalization_count+len_sample_batched)) * full_gradient +\
                    (len_sample_batched/(normalization_count+len_sample_batched)) * grad(train_loss)(flatten_params(model_params), sample_batched, remove_label=None)
            hess = (normalization_count/(normalization_count+len_sample_batched)) * hess +\
                (len_sample_batched/(normalization_count+len_sample_batched)) * hessian(train_loss)(flatten_params(model_params), sample_batched)
            normalization_count += len_sample_batched
        # for i, sample_batched in enumerate(train_loader):
        #     full_gradient += grad(train_loss)(flatten_params(model_params), sample_batched, remove_label=None)
        #     hess += hessian(train_loss)(flatten_params(model_params), sample_batched)
        hess += 1e-9 * torch.eye(hess.shape[0]).to(args.device)
        print("condition number", torch.linalg.cond(hess))
        hess_inv = torch.linalg.inv(hess)

        val_loader = DataLoader(
            dataset_val, batch_size=1, shuffle=False,
            collate_fn=collate_fn_svd)

        test_gradients = []
        for i, sample_batched in enumerate(val_loader):
            if i > 499:
                break
            test_gradient = grad(train_loss)(flatten_params(model_params), sample_batched, test=True)  # train_loss
            test_gradients.append(test_gradient)
        test_gradients = torch.stack(test_gradients, dim=0)  # (100, p)
        print("test_gradients.shape", test_gradients.shape)

        inf_thetas = []
        for remove_label in range(100): # 159
            gradient_i = 0
            normalization_count = 0
            for i, sample_batched in enumerate(train_loader):
                len_sample_batched = len(sample_batched[1])
                gradient_i = (normalization_count/(normalization_count+len_sample_batched)) * gradient_i +\
                    (len_sample_batched/(normalization_count+len_sample_batched)) * grad(train_loss)(flatten_params(model_params), sample_batched, remove_label=remove_label)
                normalization_count += len_sample_batched
                # print(normalization_count)
            # for i, sample_batched in enumerate(train_loader):
            #     gradient_i += grad(train_loss)(flatten_params(model_params), sample_batched, remove_label=remove_label)
            # print(gradient_i[1002+remove_label*2:1002+remove_label*2+2])
            # removed_range = (1002+remove_label*2, 1002+remove_label*2+2)
            # removed_ranges = [(4160+remove_label, 4160+remove_label+1),
            #                   (1616+remove_label*16, 1616+remove_label*16+16)]
            # removed_ranges = [(2168+remove_label, 2168+remove_label+1),  # delicious
            #                   (202+remove_label*2, 202+remove_label*2+2)]
            # removed_ranges = [(8672+remove_label, 8672+remove_label+1),  # delicious dim_8
            #                   (808+remove_label*8, 808+remove_label*8+8)]
            removed_ranges = [(983*8+remove_label, 983*8+remove_label+1), (remove_label*8, remove_label*8+8)]
            # removed_ranges = [(159*50+remove_label, 159*50+remove_label+1), (remove_label*50, remove_label*50+50)]


            hess_inv_delete = hess_inv.clone()
            full_gradient_delete = full_gradient.clone()
            gradient_i_delete = gradient_i.clone()
            for removed_range in removed_ranges:
                print(gradient_i_delete[removed_range[0]:removed_range[1]])
                gradient_i_delete = torch.cat([gradient_i_delete[:removed_range[0]], gradient_i_delete[removed_range[1]:]])
                hess_inv_delete = torch.cat([hess_inv_delete[:, :removed_range[0]], hess_inv_delete[:, removed_range[1]:]], dim=1)
                hess_inv_delete = torch.cat([hess_inv_delete[:removed_range[0]], hess_inv_delete[removed_range[1]:]], dim=0)
                print(hess_inv_delete.shape)
                full_gradient_delete = torch.cat([full_gradient_delete[:removed_range[0]], full_gradient_delete[removed_range[1]:]])

            print(remove_label)
            inf_theta = hess_inv_delete @ (full_gradient_delete - gradient_i_delete)
            inf_thetas.append(inf_theta)
        inf_thetas = torch.stack(inf_thetas, dim=0)  #(500, p)

        scores = []
        for remove_label in range(100):  # 159
            # removed_ranges = [(4160+remove_label, 4160+remove_label+1),
            #                   (1616+remove_label*16, 1616+remove_label*16+16)]
            # removed_ranges = [(2168+remove_label, 2168+remove_label+1),
            #                   (202+remove_label*2, 202+remove_label*2+2)]
            # removed_ranges = [(8672+remove_label, 8672+remove_label+1),  # delicious dim_8
            #                   (808+remove_label*8, 808+remove_label*8+8)]
            removed_ranges = [(983*8+remove_label, 983*8+remove_label+1), (remove_label*8, remove_label*8+8)]
            # removed_ranges = [(159*50+remove_label, 159*50+remove_label+1), (remove_label*50, remove_label*50+50)]
            test_gradients_delete = test_gradients.clone()
            for removed_range in removed_ranges:
                # print(test_gradients_delete[:, removed_range[0]:removed_range[1]])
                test_gradients_delete = torch.cat([test_gradients_delete[:, :removed_range[0]], test_gradients_delete[:, removed_range[1]:]], dim=1)
                # test_gradients_delete[removed_range[0]:removed_range[1]] = 0
            score = test_gradients_delete @ inf_thetas[remove_label].T
            scores.append(score)
        scores = torch.stack(scores, dim=1)

        torch.save(scores, f"score/score_{args.dataset}_100_500_-9_dim_8_lr_fixed_hessian_norm_fix_gradient_norm.pth")

            # del gradient
            # hess = hessian(train_loss)(flatten_params(model_params), sample_batched)
            # print(hess, hess.shape)
            # hess_inv = torch.linalg.inv(hess + 1e-3 * torch.eye(hess.shape[0]).cuda(1))
            # print(hess_inv, hess_inv.shape)
            # print(sample_batched[0][0].dtype, sample_batched[0][1].dtype)
            # ihvp = ihvp_cg(train_loss)((flatten_params(model_params), sample_batched), gradient)
            # print(ihvp, ihvp.shape)