from argparse import ArgumentError

import joblib
import numpy as np
import os

import pandas as pd

import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from src.myscripts import func


def _encode_labels(df, label_col_index, encoder_class, encoder_file_name):
    Y = df.iloc[:, label_col_index].values
    encoder = encoder_class()

    if isinstance(encoder, OneHotEncoder):
        encoder = encoder_class(sparse=False)
        Y = encoder.fit_transform(Y.reshape(-1, 1))
    elif isinstance(encoder, LabelEncoder):
        Y = encoder.fit_transform(Y)
    else:
        raise AttributeError("Unknown encoder parsed!:", encoder_class)
    dataframe = df
    dataframe.iloc[:, label_col_index] = Y
    joblib.dump(encoder, f"{config.MODEL_DIR_PATH}/{encoder_file_name}")
    return dataframe

def prepare_data_for_model(df, label_col_index, encoder_class, split_proportions, random_state=0, save_to_npy=True, encoder_file_name="saved_encoder.joblib"):
    path= config.DATA_SPLIT_SAVE_DIR_PATH
    if(round(sum(split_proportions),6)!=1):
        raise AttributeError(f"split_proportions does not add up to 1. Passed split_proportions={split_proportions}={sum(split_proportions)} (correct example: split_proportions=[0.7,0.2,0.1])")
    data_encoded = _encode_labels(df, label_col_index, encoder_class, encoder_file_name=encoder_file_name)
    X = data_encoded.iloc[:, :-1].values
    Y = data_encoded.iloc[:, -1].values

    # Podzial zbioru na dane do uczenia, zbior validacyjny oraz zbior do pozniejszego testowania danych.
    try:
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1-split_proportions[0], random_state=random_state, shuffle=True, stratify=Y)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=split_proportions[2]/(split_proportions[1]+split_proportions[2]), random_state=random_state, shuffle=True,
                                                        stratify=y_test)
    except:
        raise ArgumentError("Insufficient population of class. At least 2 representatives needed to split data.")

    # Rozszerzenie wymiarów danych, ponieważ dla CNN potrzeba 3 wymiarow (wysokosc, szerokosc i ilosc kanalow obrazu)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    if save_to_npy:
        np.save(os.path.join(path, "x_train.npy"), x_train)
        np.save(os.path.join(path, "y_train.npy"), y_train)
        np.save(os.path.join(path, "x_val.npy"), x_val)
        np.save(os.path.join(path, "y_val.npy"), y_val)
        np.save(os.path.join(path, "x_test.npy"), x_test)
        np.save(os.path.join(path, "y_test.npy"), y_test)
        print(f"Prepared data saved in: {path}.\nData shape confirmation:"
              f"\nx_train:{x_train.shape}"
              f"\ny_train:{y_train.shape}"
              f"\nx_val:{x_val.shape}"
              f"\ny_val:{y_val.shape}"
              f"\nx_test:{x_test.shape}"
              f"\ny_test:{y_test.shape}"
              f"\n\nNumber of classes: {len(np.unique(y_train))}")
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test


