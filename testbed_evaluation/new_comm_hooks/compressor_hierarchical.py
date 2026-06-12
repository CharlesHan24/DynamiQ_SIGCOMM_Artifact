import torch
from torch.nn.functional import pad
import numpy as np
import pdb
import time
import sys
import math
import ctypes

class NewINCACompressor(object):
    def __init__(self, params):
        # self.use_compressor_list = params.get('use_compressor_list', False)
        # eden_utils.Hadamard_init()
        
        self.device = params.get('device', 'cuda') 
        

        self.ds = params['d']
        self.original_size = params["size"]
        self.to_rescale = params.get("to_rescale", True)
        
        self.seed = params.get('seed', 42)
        self.nclients = params.get('nclients', 1)

        self.max_chunk_size = params.get("chunk_size", 0)
        self.compress_vec = dict()
        self.max_memory = dict()
        self.padded_tensor = dict()
        for name_idx in self.ds:
            dim = self.original_size[name_idx]
            self.padded_tensor[name_idx] = torch.zeros(((self.ds[name_idx] + self.max_chunk_size - 1) // self.max_chunk_size * self.max_chunk_size), dtype=torch.bfloat16, device=self.device)
            self.max_memory[name_idx] = torch.zeros(((self.ds[name_idx] + self.max_chunk_size - 1) // self.max_chunk_size), dtype=torch.bfloat16, device=self.device)


    def padding_tensor(self, tensor, name):
        orig_size = dim = padded_dim = self.original_size[name]
        torch.div(tensor, self.nclients, out=self.padded_tensor[name][:dim])
        
        return self.padded_tensor[name], self.max_memory[name]

    def max_memory_tensor(self, name):
        return self.max_memory[name]
