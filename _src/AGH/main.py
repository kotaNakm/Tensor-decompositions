import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse
import os
import shutil
from AGH import AGH

import sys
sys.path.append("_src")
import utils



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input/Output
    parser.add_argument("--input_tag", type=str)  #
    parser.add_argument("--input_type", type=str)  #
    parser.add_argument("--out_dir", type=str)  #
    parser.add_argument("--entities", type=str)
    parser.add_argument("--value_column", type=str)
    parser.add_argument("--import_type", type=str, default="clean_data")

    # model
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--initial_gamma", type=float, default=0.01)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument(
        "--optimization", type=str, choices=["full", "wo_gradient_ascent","wo_adaptive_steps"], default="full"
    )
    parser.add_argument("--negative_curvature", type=float, default=1)

    args = parser.parse_args()

    # I/O setup
    outputdir = args.out_dir
    if os.path.exists(outputdir):
        shutil.rmtree(outputdir)
    os.makedirs(outputdir)

    entities = args.entities.split("/")
    value_column = args.value_column

    raw_df = utils.import_dataframe(args)
    encoded_df = utils.prepare_tensor(
        raw_df,
        entities,
        value_column
    )

    # Model configuration
    phai = encoded_df[value_column].max()

    tensor_shape = (encoded_df[entities].max()+1).values.astype(int)
    encoded_tensor = encoded_df.to_numpy()
    
    # TODO:randmize train_tensor vs test tensor

    train_len=int(len(encoded_tensor)*0.01)
    train_tensor = encoded_tensor[:train_len]
    test_tensor = encoded_tensor[train_len:]
    
    
    agh = AGH(
        tensor_shape,
        args.rank,
        args.initial_gamma,
        args.q,
        args.negative_curvature,
        args.optimization,
        phai,
    )

    factors = agh.train(
        train_tensor,
    )
