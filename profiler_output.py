import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
import numpy as np
import logging
from torch.profiler import ExecutionTraceObserver
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from typing import Optional, Union
from torch.profiler import ExecutionTraceObserver

class DNN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout()

        self.fc2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()

        self.fc3 = nn.Linear(64, output_dim)
        self.act3 = nn.Softmax(dim=-1)

    def forward(self, x):
        b, t, c = x.size()
        x = x.reshape(b, -1)
        out = self.dropout(self.act1(self.fc1(x)))
        out = self.act2(self.fc2(out))
        out = self.act3(self.fc3(out))

        return out
    
#Reading the example data 
path = 'data/subject101.dat' # set this to path of dat file
data = []
with open(path, 'r') as f:
    #transform dat into python list
    d = f.readlines()
    for i in d:
        k = i.rstrip().split(" ")
        data.append([float(i) for i in k]) 
a = torch.tensor([data])

model_dnn = DNN(20326518, 1)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model_dnn(a)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    model_dnn(a)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))