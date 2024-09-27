import os
import numpy as np
import pandas as pd
import tqdm


def load_data(input_type, input_format):
    filepath = os.path.dirname(__file__)
    df = pd.read_csv(filepath + "project_tycho.csv.gz")
    

    # preprocess type
    if input_type=="ver1":
        # preprocess the data TODO

    # reformat data
    if input_format=="numpy":
        data = 0
    elif input_format=="pandas":
        data = 0


    return data