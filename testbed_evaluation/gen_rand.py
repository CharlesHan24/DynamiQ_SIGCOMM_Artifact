import numpy as np
import torch
import pickle
import random


def generate_random_sampling_order(nsamples, nranks, nepochs):
    nsamples = nsamples // nranks * nranks
    all_orders = [[] for i in range(nranks)]
    for i in range(nepochs):
        orders = [i for i in range(nsamples)]
        random.shuffle(orders)
        for j in range(nranks):
            all_orders[j].append(orders[j::nranks])
    
    pickle.dump(all_orders, open("./models/indices_{}_{}.pkl".format(nranks, nepochs), "wb"))

generate_random_sampling_order(51200, 8, 3)
generate_random_sampling_order(51200, 2, 3)