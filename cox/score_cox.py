import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric, support
from pycox.models import CoxPH

from torch import nn
import argparse

from pycox.models.loss import cox_ph_loss
from dattri.func.utils import flatten_func, flatten_params
from torch.func import grad, hessian, vmap

class LinearModelNoBias(nn.Module):
  def __init__(self, input_features, output_features):
    super(LinearModelNoBias, self).__init__()
    # Define a linear layer without bias
    self.linear = nn.Linear(input_features, output_features, bias=False)

  def forward(self, x):
    # Pass input through the linear layer
    return self.linear(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CoxPH model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dataset', type=str, default='metabric', help='Dataset to use')
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    _ = torch.manual_seed(args.seed)

    # prepare and preprocess data
    if args.dataset == 'metabric':
      df_train = metabric.read_df()
    if args.dataset == 'support':
      df_train = support.read_df()
    df_test = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_test.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)

    if args.dataset == 'metabric':
      cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
      cols_leave = ['x4', 'x5', 'x6', 'x7']
    elif args.dataset == 'support':
      cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
      cols_leave = ['x1', 'x4', 'x5']

    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [(col, None) for col in cols_leave]

    x_mapper = DataFrameMapper(standardize + leave)

    x_train = x_mapper.fit_transform(df_train).astype('float32')
    x_val = x_mapper.transform(df_val).astype('float32')
    x_test = x_mapper.transform(df_test).astype('float32')

    get_target = lambda df: (df['duration'].values, df['event'].values)
    y_train = get_target(df_train)
    y_val = get_target(df_val)
    durations_test, events_test = get_target(df_test)
    val = x_val, y_val

    # create a linear model without bias
    in_features = x_train.shape[1]
    out_features = 1
    net = LinearModelNoBias(in_features, out_features)  # linear model
    model = CoxPH(net, tt.optim.Adam)  # coxph model
    model.optimizer.set_lr(0.01)

    # load the fully trained model
    model.load_net(f'./checkpoints_{args.dataset}_full_batch/linear_model_seed_{args.seed}_remove_index_-1')
    model.net.cpu()

    model_params = {k: p for k, p in model.net.named_parameters() if p.requires_grad}

    @flatten_func(model.net)
    def loss_func(params, data, remove_index):
        x, (y_0, y_1) = data
        if remove_index is not None:
            x = torch.cat((x[:remove_index], x[remove_index+1:]))
            y_0 = torch.cat((y_0[:remove_index], y_0[remove_index+1:]))
            y_1 = torch.cat((y_1[:remove_index], y_1[remove_index+1:]))
        yhat = torch.func.functional_call(model.net, params, x)
        return cox_ph_loss(yhat, y_0, y_1)

    @flatten_func(model.net)
    def predict_re_haz(param, data):
        return torch.exp(torch.func.functional_call(model.net, param, data))[0]

    scores = []
    grad_test = vmap(grad(predict_re_haz), in_dims=(None, 0))(flatten_params(model_params),
                                                              torch.tensor(x_test))
    grad_train_full = grad(loss_func)(flatten_params(model_params), (torch.tensor(x_train),
                                                                     (torch.tensor(y_train[0]), torch.tensor(y_train[1]))), None)
    hessian_train_full_inv = torch.linalg.inv(hessian(loss_func)(flatten_params(model_params),
                                                                 (torch.tensor(x_train),
                                                                  (torch.tensor(y_train[0]), torch.tensor(y_train[1]))), None))
    for i in range(x_train.shape[0]):  # x_train.shape[0]
        if i % 100 == 0:
            print(i)
        score = grad_test @ hessian_train_full_inv @ (grad_train_full - grad(loss_func)(flatten_params(model_params),
                                                                                        (torch.tensor(x_train), (torch.tensor(y_train[0]), torch.tensor(y_train[1]))), i))
        scores.append(score)
    scores = torch.stack(scores, dim=0)

    torch.save(scores, f'scores_seed_{args.seed}_full_{args.dataset}.pt')