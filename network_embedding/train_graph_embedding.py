import matplotlib.pyplot as plt
import networkx as nx

import torch
import torch.nn as nn
import random
from tqdm import tqdm

import numpy as np
import random

import argparse

import time

w=3            # window size
d=2            # embedding size
y=1000          # walks per vertex  # normally 400
t=6            # walk length
lr=0.025       # learning rate

argparser = argparse.ArgumentParser()
argparser.add_argument('--remove_index', type=int, default=-1)
args = argparser.parse_args()

st = time.time()

G = nx.karate_club_graph()
size_vertex = G.number_of_nodes()  # number of vertices

if args.remove_index != -1:
    G.remove_node(args.remove_index)
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
    return torch.log(torch.sum(torch.exp(out))) - out[target]

def skip_gram(wvi,  w, model):
    for j in range(len(wvi)):
        for k in range(max(0,j-w) , min(j+w, len(wvi))):
            #generate one hot vector
            one_hot          = torch.zeros(size_vertex)
            one_hot[wvi[j]]  = 1

            out              = model(one_hot)
            if args.remove_index != -1:
                out[args.remove_index] = -torch.inf
            loss             = loss_fn(out, wvi[k])
            loss.backward()

            # update the parameters with SGD
            for param in model.parameters():
                param.data.sub_(lr*param.grad)
                param.grad.data.zero_()

if __name__ == '__main__':
    for seed in range(10):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = Model()

        # Train a full model
        for i in tqdm(range(y)):
            random.shuffle(v)
            for vi in v:
                path=RandomWalk(vi, t, G)
                skip_gram(path, w, model)

        # prompt: get the label of nodes in G
        G_new = nx.karate_club_graph()
        labels = nx.get_node_attributes(G_new, 'club')
        labels_list = []
        for key, value in labels.items():
            if value == 'Mr. Hi':
                labels_list.append(0)
            else:
                labels_list.append(1)

        plt.figure()
        plt.scatter(model.phi.data[:,0], model.phi.data[:,1], c=labels_list)
        for i in range(size_vertex):
            plt.annotate(i, (model.phi.data[i,0], model.phi.data[i,1]))

        print(time.time() - st)

        plt.savefig(f'images/karate_club_embedding_full_remove_{args.remove_index}_1000_seed_{seed}.png')
        torch.save(model.state_dict(), f'checkpoints_deepwalk_1000/model_karate_club_remove_{args.remove_index}_seed_{seed}.pth')
