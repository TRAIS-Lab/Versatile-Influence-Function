import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import random
from tqdm import tqdm

from torch.func import grad, hessian
from dattri.func.utils import flatten_params, flatten_func
from pathlib import Path

import time

w=3            # window size
d=2            # embedding size
y=400          # walks per vertex
t=6            # walk length
lr=0.025       # learning rate

st = time.time()

G = nx.karate_club_graph()
size_vertex = G.number_of_nodes()  # number of vertices
v = list(G.nodes)

def RandomWalk(node, t, graph):
    walk = [node]
    # walk length is t
    for _ in range(t-1):
        neighbors = list(graph.neighbors(walk[-1]))
        # if there is no neighbor, then repeat the last node
        if len(neighbors) == 0:
            walk.append(walk[-1])
            continue
        current = random.choice(neighbors)
        walk.append(current)
    return walk

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi  = nn.Parameter(torch.rand((size_vertex, d), requires_grad=True))
        self.phi2 = nn.Parameter(torch.rand((d, size_vertex), requires_grad=True))

    def forward(self, one_hot):
        hidden = torch.matmul(one_hot, self.phi)
        out    = torch.matmul(hidden, self.phi2)
        return out

def loss_fn(out, target):
    return torch.log(torch.sum(torch.exp(out))) - target

if __name__ == '__main__':
    model = Model()
    # model_full = Model()

    gt_list = []
    # remove_index_list = (34)
    for checkpoint_index in range(0, 34):
        # gt_item_sum = 0
        gt_item_list = []
        # seed (10)
        for seed in range(1):
            gt_item = []
            checkpoint_path = Path("checkpoints_deepwalk_1000") / f"model_karate_club_remove_{checkpoint_index}_seed_{seed}.pth"

            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}
            model.eval()

            @flatten_func(model)
            def f(params, vertex_pair, remove_index):
                j, k = vertex_pair

                # prepare the one hot vector
                one_hot = torch.zeros(size_vertex)
                one_hot[j]  = 1

                yhat = torch.func.functional_call(model, params, one_hot)
                target = yhat[k]
                if remove_index is not None:
                    yhat = torch.cat((yhat[:remove_index], yhat[remove_index+1:]))

                return loss_fn(yhat, target)

            # test samples (34, 34)
            for j in range(1,35):
                gt_item_item = []
                for k in range(1,35):
                    gt_item_item.append(f(flatten_params(model_params), (j-1, k-1), checkpoint_index))
                gt_item.append(torch.tensor(gt_item_item))
            
            print(checkpoint_index)

            # gt_item_sum += torch.stack(gt_item)
            gt_item_list.append(torch.stack(gt_item))
        
        # gt_list.append(gt_item_sum)
        gt_list.append(torch.stack(gt_item_list))
        
    gt = torch.stack(gt_list)

    print(gt.shape)
    print(time.time() - st)
    torch.save(gt, "gt_karate_club_rm_embedding_10_seperate_1000.pt")



    