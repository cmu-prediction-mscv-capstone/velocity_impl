from modules.train.train_lva import Train
import logging as Logger
from data.dataloader import TrajectoryDataLoader
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
from modules.model.lva import LVAttNet
from torch.optim import lr_scheduler
from torch.autograd import Variable
from modules.train.pytorchtools import EarlyStopping
from tputils.utils import get_ade,get_fde,trajectory_matrix_norm,make_trajectories_array
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train velocity network.')
    parser.add_argument('--data_dir', type=str, default='../../datasets/univ/')
    parser.add_argument('--obs_len', type=int,default=8)
    parser.add_argument('--pred_len', type=int,default=12)
    parser.add_argument('--normalize_type',type=int,default=0)
    parser.add_argument('--img_width',type=int,default=0)
    parser.add_argument('--img_height',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--validation_split',type=float,default=0.2)

    args=parser.parse_args()
    train_object=Train(args.data_dir,args.obs_len,args.pred_len,args.normalize_type,args.img_width,
                        args.img_height,args.batch_size,args.validation_split)
    train_object.build_dataloader()
    train_object.load_network(emb_dim=256, hidden_dim=256, dropout=0, lr=0.001, step_size=1000, gamma=0.1, output_dim=2)
    train_object.train(epochs=30, save_path='../saved_models')