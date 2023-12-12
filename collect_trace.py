import pandas as pd
import numpy as np
import torch.profiler
import logging
from torch.profiler import ExecutionTraceObserver
import numpy as np
import torch
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


def test_execution_trace_start_stop(self):
    et = ExecutionTraceObserver()
    et.register_callback("pytorchET.json")
    et.start()
    model_dnn(a)
    et.stop()
    et.unregister_callback()

test_execution_trace_start_stop(None)

def trace_handler(prof):
    prof.export_chrome_trace("kinetoET.json")

prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU
                    #, torch.profiler.ProfilerActivity.CUDA
                    ],
        schedule=torch.profiler.schedule(wait=0, warmup=2, active=1),
        record_shapes=True,
        on_trace_ready=trace_handler,
        with_stack=True,
        profile_memory=True,
        #with_flops=True,
        #with_modules=True
        )
prof.start()
model_dnn(a)
prof.stop()
trace_handler(prof)