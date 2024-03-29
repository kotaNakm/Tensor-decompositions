import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse
import os
import shutil
import random
import dill
import torch
import torch.nn.functional as F


import sys
sys.path.append("factorization")

import utils
from parafac import PARAFAC

seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument("--input_tag", type=str)  #
    parser.add_argument("--input_type", type=str)  #
    parser.add_argument("--out_dir", type=str)  #
    parser.add_argument("--entities", type=str)
    parser.add_argument("--value_column", type=str)
    parser.add_argument("--import_type", type=str, default="clean_data")

    # Model
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--optim",type=str,default="aaaaaaa") #TODO
    parser.add_argument("--lr",type=float,default=1e-3) #TODO


    # Experimenantal setup
    parser.add_argument("--train_ratio", type=float, default=0.7)

    args = parser.parse_args()

    # I/O setup
    outputdir = args.out_dir
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    entities = args.entities.split("/")
    value_column = args.value_column

    raw_df = utils.import_dataframe(args)
    encoded_df = utils.prepare_tensor(raw_df, entities, value_column)
    tensor_shape = (encoded_df[entities].max() + 1).values.astype(int)
    nonzero_records = encoded_df.to_numpy()

    nonzero_records_train, nonzero_records_test = utils.train_test_split(nonzero_records, args.train_ratio)
    train_tensor = utils.nz_records_to_tensor(nonzero_records_train, tensor_shape)
    
    parafac = PARAFAC(tensor_shape, args.rank)
    for a in parafac.parameters():
        print(a)

    optimizer = torch.optim.Adagrad(parafac.parameters(), lr=args.lr)
    factors = parafac.training_tensor(train_tensor, args.n_iter, optimizer)
    re_tensor = parafac.forward().detach().numpy()
    erorr = (train_tensor - re_tensor)**2
    print(erorr)
    print(outputdir)
    # np.save(f"{args.out_dir}/loss_logs", loss_logs)
    # dill.dump([factors,loss_logs], open(f"{outputdir}/result.dill", "wb"))
