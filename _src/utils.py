import numpy as np
import pandas as pd
from sklearn import preprocessing
from importlib import import_module

import sys
sys.path.append("_dat")


def import_dataframe(args):
    input_tag = args.input_tag
    input_type = args.input_type

    dataset_module = import_module(input_tag)

    # For import of synthetic data
    if input_type == "AGH_experiment1":
        return dataset_module.AGH_experiment1()

    else:
        NotImplementedError


def prepare_tensor(given_data, entities, value_column):
    data = given_data.copy("deep")
    data = data.dropna(subset=entities)

    # Encode entities
    for tmp_idx in entities:
        print(tmp_idx)
        print(data.loc[:, tmp_idx])
        le = preprocessing.LabelEncoder()
        # data[tmp_idx] = data[tmp_idx].astype("str")
        data[tmp_idx] = le.fit_transform(data[tmp_idx])
        data[tmp_idx] = data[tmp_idx].astype(int)

    return data
