import pickle
from utils import Corpus,  batchify, word_dict
import argparse

from model_loc import *
import torch.nn as nn
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=64)
args = parser.parse_args()

net = torch.load('./data/out/subj_no_norm_radius_dim_d_update_init.pkl')
if torch.cuda.is_available():
    device = 'cuda:3'
else:
    device = 'cpu'
device = torch.device(device)
# word
# i 1005
# shirt 8005
# address 1915
# pen 16844
# book 1317
# cat 1357
# figure 1311
# you 1130
# yours 20448
# dog 1337
# like 46
# mine 3402

# _cls_ + word
sw = torch.tensor([[1,1005]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print(x.shape)
print(init_phase.shape)
print(period.shape)
print("i : ", init_radius[0,1].mean())
print("i : ", period[0,1].abs().mean())


sw = torch.tensor([[1,8005]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("shirt : ", init_radius[0,1].mean())
print("shirt : ", period[0,1].abs().mean())


sw = torch.tensor([[1,1130]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("you : ", init_radius[0,1].mean())
print("you : ", period[0,1].abs().mean())

sw = torch.tensor([[1,20448]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("your : ", init_radius[0,1].mean())
print("your : ", period[0,1].abs().mean())

sw = torch.tensor([[1,1915]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("address : ", init_radius[0,1].mean())
print("address : ", period[0,1].abs().mean())

sw = torch.tensor([[1,1311]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("figure : ", init_radius[0,1].mean())
print("figure : ", period[0,1].abs().mean())

sw = torch.tensor([[1,46]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("like : ", init_radius[0,1].mean())
print("like : ", period[0,1].abs().mean())


sw = torch.tensor([[1,16844]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("pen : ", init_radius[0,1].mean())
print("pen : ", period[0,1].abs().mean())

sw = torch.tensor([[1,1317]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("book : ", init_radius[0,1].mean())
print("book : ", period[0,1].abs().mean())

sw = torch.tensor([[1,1357]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("cat : ", init_radius[0,1].mean())
print("cat : ", period[0,1].abs().mean())


sw = torch.tensor([[1,1337]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("dog : ", init_radius[0,1].mean())
print("dog : ", period[0,1].abs().mean())

sw = torch.tensor([[1,3402]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("mine : ", init_radius[0,1].mean())
print("mine : ", period[0,1].abs().mean())


sw = torch.tensor([[1,288]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("good : ", init_radius[0,1].mean())
print("good : ", period[0,1].abs().mean())

sw = torch.tensor([[1,1139]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("bad : ", init_radius[0,1].mean())
print("bad : ", period[0,1].abs().mean())

sw = torch.tensor([[1,14552]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("dislike : ", init_radius[0,1].mean())
print("dislike : ", period[0,1].abs().mean())

sw = torch.tensor([[1,834]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("do : ", init_radius[0,1].mean())
print("do : ", period[0,1].abs().mean())


sw = torch.tensor([[1,2122]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("doing : ", init_radius[0,1].mean())
print("doing : ", period[0,1].abs().mean())


sw = torch.tensor([[1,834]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("do : ", init_radius[0,1].mean())
print("do : ", period[0,1].abs().mean())


sw = torch.tensor([[1,2122]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("doing : ", init_radius[0,1].mean())
print("doing : ", period[0,1].abs().mean())


sw = torch.tensor([[1,10]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("in : ", init_radius[0,1].mean())
print("in : ", period[0,1].abs().mean())


sw = torch.tensor([[1,136]]).to(device)
x, init_radius, period, init_phase = net.bert(sw, None)
print("on : ", init_radius[0,1].mean())
print("on : ", period[0,1].abs().mean())