import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse
import os
import shutil
import random
import dill
import pickle

import sys

sys.path.append("_src")
import utils
from agh import AGH

random.seed(100)

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
    parser.add_argument("--l0", type=float, default=0.08)
    parser.add_argument("--L", type=float, default=0.1)
    parser.add_argument("--n_iter", type=int, default=30)

    parser.add_argument(
        "--optimization",
        type=str,
        choices=["full", "wo_gradient_ascent", "wo_adaptive_steps"],
        default="full",
    )
    parser.add_argument("--negative_curvature", type=float, default=1)

    # experimenantal setup
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

    # Model configuration
    phai = encoded_df[value_column].max()

    tensor_shape = (encoded_df[entities].max() + 1).values.astype(int)
    encoded_tensor = encoded_df.to_numpy()

    shuffuled_idxs = random.sample(
        list(range(len(encoded_tensor))), len(encoded_tensor)
    )
    shuffuled_tensor = encoded_tensor[shuffuled_idxs]
    train_len = int(len(shuffuled_tensor) * args.train_ratio)
    train_tensor = shuffuled_tensor[:train_len]
    test_tensor = shuffuled_tensor[train_len:]

    agh = AGH(
        tensor_shape,
        args.rank,
        args.initial_gamma,
        args.l0,
        args.L,
        args.negative_curvature,
        args.optimization,
        phai,
        args.n_iter,
    )

    factors, loss_logs = agh.train(
        train_tensor,
    )

    # np.save(f"{args.out_dir}/loss_logs", loss_logs)
    with open(f"{outputdir}/result.pkl","wb") as f:
        pickle.dump(agh,f)
    print(outputdir)
