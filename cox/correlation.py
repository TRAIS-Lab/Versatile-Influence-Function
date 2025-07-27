from scipy.stats import pearsonr, spearmanr
from scipy.stats import combine_pvalues
import torch
import numpy as np

if __name__ == '__main__':
    scores = torch.load("score/scores_seed_0_full_metabric.pt")
    gt = torch.load("score/gt_cox_seed_1_full_metabric.pt")

    print(scores.shape, gt.shape)

    pearson_corr_list = []
    pearson_pval_list = []
    for i in range(scores.shape[1]):
        pearson_corr, pvalue = pearsonr(scores[:, i].detach().numpy(),
                                gt[:, i].detach().numpy())
        if not np.isnan(pearson_corr):
            pearson_corr_list.append(pearson_corr)
            pearson_pval_list.append(pvalue)
    print("pearson mean:", np.mean(pearson_corr_list))
    print("pearson pval:", combine_pvalues(pearson_pval_list))

    spearman_corr_list = []
    spearman_pval_list = []
    for i in range(scores.shape[1]):
        spearman_corr, pvalue = spearmanr(scores[:, i].detach().numpy(),
                                gt[:, i].detach().numpy())
        if not np.isnan(spearman_corr):
            spearman_corr_list.append(spearman_corr)
            spearman_pval_list.append(pvalue)
    print("spearman mean:", np.mean(spearman_corr_list))
    print("spearman pval:", combine_pvalues(spearman_pval_list))
