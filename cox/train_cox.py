import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from pycox.datasets import metabric, support
from pycox.models import CoxPH

from torch import nn
import argparse

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
    parser.add_argument('--remove_index', type=int, default=-1, help='Index to remove from the dataset')
    parser.add_argument('--dataset', type=str, default='metabric', help='Dataset to use')
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    _ = torch.manual_seed(args.seed)

    # prepare and preprocess data
    if args.dataset == 'metabric':
      df_train = metabric.read_df()
    elif args.dataset == 'support':
      df_train = support.read_df()
    else:
      raise ValueError('Invalid dataset')
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

    # remove index
    print("Removing index:", args.remove_index)
    if args.remove_index != -1:
        x_train = np.delete(x_train, args.remove_index, axis=0)
        y_train = (np.delete(y_train[0], args.remove_index), np.delete(y_train[1], args.remove_index))

    print("x_train.shape:", x_train.shape)
    print("y_train[0].shape:", y_train[0].shape)
    print("y_train[1].shape:", y_train[1].shape)

    # create a linear model without bias
    in_features = x_train.shape[1]
    out_features = 1
    if args.dataset == "metabric":
      batch_size = 2000
    if args.dataset == "support":
      batch_size = 6000
    net = LinearModelNoBias(in_features, out_features)  # linear model
    print(net)
    model = CoxPH(net, tt.optim.Adam)  # coxph model
    model.optimizer.set_lr(0.01)

    if args.dataset == "metabric":
      epochs = 200
    if args.dataset == "support":
      epochs = 100

    callbacks = []
    verbose = True
    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose,
                    val_data=val, val_batch_size=batch_size)
    _ = model.compute_baseline_hazards()

    model.save_net(f'./checkpoints_support_full_batch/linear_model_seed_{args.seed}_remove_index_{args.remove_index}')
