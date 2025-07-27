import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import random
from tqdm import tqdm

from torch.func import grad, hessian
from dattri.func.utils import flatten_params, flatten_func

import argparse


w=3            # window size
d=2            # embedding size
y=1000          # walks per vertex
t=6            # walk length
lr=0.025       # learning rate

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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=0)
    args = argparser.parse_args()

    model = Model()

    model.load_state_dict(torch.load(f'checkpoints_deepwalk_1000/model_karate_club_remove_-1_seed_{args.seed}.pth', map_location='cpu'))
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


    model_params = {k: p for k, p in model.named_parameters() if p.requires_grad}

    # get the walk here
    walks = {}
    for remove_index in [None] + list(range(34)):
        walk = []
        G_new = G.copy()
        if remove_index is not None:
            G_new.remove_node(remove_index)
        v_new = list(G_new.nodes)

        for i in tqdm(range(y)):
            random.shuffle(v_new)
            for vi in v_new:
                path=RandomWalk(vi, t, G_new)
                for j in range(len(path)):
                    for k in range(max(0,j-w) , min(j+w, len(path))):
                        walk.append((path[j], path[k]))
        walks[remove_index] = walk
        print(len(walks[remove_index]))

    gradient_full = 0
    hess = 0
    grad_val_list = []
    with torch.no_grad():
        for walk in tqdm(walks[None]):
            gradient_full += grad(f)(flatten_params(model_params), walk, None)
            hess += hessian(f)(flatten_params(model_params), walk, None)
        for remove_index in list(range(34)):
            grad_val = 0
            for walk in tqdm(walks[remove_index]):
                grad_val += grad(f)(flatten_params(model_params), walk, remove_index)
            grad_val_list.append(grad_val)

    torch.save(grad_val_list, f'grad_val_list_seed_{args.seed}_1000.pt')
    torch.save(hess, f'h_val_seed_{args.seed}_1000.pt')
    torch.save(gradient_full, f'gradient_full_seed_{args.seed}_1000.pt')
