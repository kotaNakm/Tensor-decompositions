import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import random

from importlib import import_module
import sys
sys.path.append("_dat")

def import_dataframe(args):
    input_tag = args.input_tag
    input_type = args.input_type

    # For import of synthetic data
    if input_tag=="synthetic":
        return generate_synthetic_data(input_type)
    else:
        dataset_module = import_module(input_tag)
        if input_type == "AGH_experiment1":
            return dataset_module.AGH_experiment1()
        else:
            NotImplementedError

def generate_synthetic_data(string_mode_dims, n_elements=3000, param=(-1,1)):
    raw_df = pd.DataFrame()
    mode_dims = [int(s) for s in string_mode_dims.split(",")]
    attributes_names= []
    for i, n_dim in enumerate(mode_dims):
        values = np.random.randint(n_dim, size=n_elements)
        att_name = f"entity{i+1}"
        raw_df[att_name] = values
        attributes_names.append(att_name)
    raw_df["value"] = np.random.normal(param[0],param[1],n_elements)
    
    raw_df = raw_df.groupby(attributes_names).agg(**{"value":("value","sum")}).reset_index()
    print(raw_df)
    return raw_df

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


def train_test_split(df_numpy, train_ratio):
    shuffuled_idxs = random.sample(
        list(range(len(df_numpy))), len(df_numpy)
    )
    shuffuled_df_numpy = df_numpy[shuffuled_idxs]
    train_len = int(len(df_numpy) * train_ratio)
    train_df = df_numpy[:train_len]
    test_df = df_numpy[train_len:]
    return train_df, test_df

def nz_records_to_tensor(nonzero_records,tensor_shape):    
    """
    nonzero_records:
    shape is # of records x (# of entities + 1) 
    """
    tensor = np.zeros(tensor_shape)
    for record in nonzero_records:
        indices = record[:-1].astype(int) 
        value = record[-1]
        tensor[tuple(indices)] = value
    return tensor 

def nz_records_to_records(nonzero_records, tensor_shape):
    pass

