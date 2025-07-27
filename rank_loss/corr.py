from scipy.stats import pearsonr, spearmanr
from scipy.stats import combine_pvalues
import torch
import numpy as np

if __name__ == '__main__':
    # scores = torch.load("score_1000_300_-5_new_new.pth")[:100, :].cpu().T
    # gt = torch.load("ground_truth/loss_seed_0.pth")[:, :100].cpu()
    # scores = torch.load("score/score_Delicious_100_500_-7_dim_8_lr_fixed_hessian_norm_fix_gradient_norm.pth")[:500, :100].cpu().T
    # gt = torch.load("ground_truth_bibtex/loss_lr_Delicious_100_500_seed_0_dim_8_relu_fix_rm.pth")[:30, :500].cpu()
    # scores = torch.load("ground_truth_bibtex/loss_lr_delicious_50_100_seed_1_dim_8_relu.pth")[:30, :500].cpu()
    # scores = torch.load("score/score_Bibtex_159_1000_-7_dim_50_lr_fixed_hessian_norm_fix_gradient_norm_fix_test_epoch_30.pth")[:823, :159].cpu().T
    # gt = torch.load("ground_truth_bibtex/loss_lr_Bibtex_159_1000_seed_0_dim_50_relu_fix_rm.pth")[:159, :823].cpu()
    # gt = torch.load("ground_truth_bibtex/loss_lr_Bibtex_159_500_seed_1_dim_50_relu_fix_rm.pth")[:159, :100].cpu()
    scores = torch.load("score/score_Mediamill2_101_1000_-5_dim_8_lr_fixed_hessian_norm_fix_gradient_norm_epoch_20.pth")[:500, :101].cpu().T
    # gt = torch.load("ground_truth_bibtex/loss_lr_Mediamill2_101_1000_seed_0_dim_8_relu_fix_rm.pth")[:101, :500].cpu()
    gt = torch.load("ground_truth_bibtex/loss_lr_Mediamill2_101_1000_seed_1_dim_8_relu_fix_rm.pth")[:101, :500].cpu()
    # scores = torch.load("ground_truth/loss_logit_seed_1.pth")[:, :100].cpu()

    print(scores.shape, gt.shape)

    pearson_corr_list = []
    pearson_pval_list = []
    for i in range(scores.shape[1]):
        if scores[:, i].sum() == 0:
            continue
        index_gt_nonnan = (gt[:, i] != torch.inf)
        # print(index_gt_nonnan.sum())
        score = scores[index_gt_nonnan, i]
        gt_ = gt[index_gt_nonnan, i]
        pearson_corr, pvalue = pearsonr(score.detach().numpy(),
                                gt_.detach().numpy())
        if not np.isnan(pearson_corr):
            pearson_corr_list.append(pearson_corr)
            pearson_pval_list.append(pvalue)
    print("pearson mean:", np.mean(pearson_corr_list))
    print("pearson pval:", combine_pvalues(pearson_pval_list))

    spearman_corr_list = []
    spearman_pval_list = []
    for i in range(scores.shape[1]):
        if scores[:, i].sum() == 0:
            continue
        index_gt_nonnan = (gt[:, i] != torch.inf)
        score = scores[index_gt_nonnan, i]
        gt_ = gt[index_gt_nonnan, i]
        spearman_corr, pvalue = spearmanr(score.detach().numpy(),
                                gt_.detach().numpy())
        if not np.isnan(spearman_corr):
            spearman_corr_list.append(spearman_corr)
            spearman_pval_list.append(pvalue)
    print("spearman mean:", np.mean(spearman_corr_list))
    print("spearman pval:", combine_pvalues(spearman_pval_list))

    print(len(pearson_corr_list))