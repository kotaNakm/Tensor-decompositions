import numpy as np
import pandas as pd
from sklearn import preprocessing

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

def generate_synthetic_data(string_mode_dims,n_elements=300, param=(0,100)):
    raw_df = pd.DataFrame()
    mode_dims = [int(s) for s in string_mode_dims.split(",")]
    for i, n_dim in enumerate(mode_dims):
        values = np.random.randint(n_dim, size=n_elements)
        raw_df[f"entity{i+1}"] = values
    raw_df["value"] = np.random.normal(param[0],param[1],n_elements)
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


def training_tensors_torch(model, data, n_iter, optimizer):
    indices = data[:,:-1]
    values = data[:,-1]
    # print(self.parameters())

    for index, value, it in zip(indices, values, np.arange(n_iter)):
        print(index)
        output = model.forward(index)
        loss_out = model.loss(value, output)
        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()
    return model.factors