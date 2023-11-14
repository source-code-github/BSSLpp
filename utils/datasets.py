import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
import os

def get_data(name):
    ''' get dataset
    Args:
        name: the name of dataset
    
    Return:
        data: the set of samples
        labels: the labels of samples in data

    '''
    if name in ['mm', 'hdc']:
        return get_data_uci(name)
    else:
        return get_data_openml(name)


def get_data_openml(name):
    ''' get dataset from openml
    Args:
        name: the name of dataset
    
    Return:
        data: the set of samples
        labels: the labels of samples in data

    '''
    datasets = {
        'aus': 40981,
        'hms': 43,
        'hep': 55,
        'kvk': 3,
        'pid': 37,
        'vote': 56,
        'wdbc': 1510,

    }

    dataset = fetch_openml(as_frame=True, data_id=datasets[name], parser="pandas")  # australian

    labels_train = dataset.target
    df = dataset.data

    for col_name in df.columns:
        if str(df[col_name].dtype) == 'category' or str(df[col_name].dtype) == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    nan_indexes = np.array(df[df.isna().any(axis=1)].index)

    if len(nan_indexes) != 0:
        df = data_process(df)

    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels_train)

    data = df.to_numpy()

    print('\nDataset: ', name)
    print('The number of classes: ', len(np.unique(labels, return_counts=True)[0]))
    samples_per_class = [f'class {i}: {samples}' for i, samples in enumerate(np.unique(labels, return_counts=True)[1])]
    print('The number of samples per class: \n', ', '.join(samples_per_class), '\n')

    return data, labels


def get_data_uci(name):
    ''' get dataset from db folder
    Args:
        name: the name of dataset
    
    Return:
        data: the set of samples
        labels: the labels of samples in data

    '''
    if name == 'hdc':
        data_path = 'db/processed.cleveland.data'
        n_f = 14
    elif name == 'mm':
        data_path = 'db/mammographic_masses.data'
        n_f = 6

    full_data_path = os.path.join(os.path.dirname(__file__), data_path)

    pd_data = pd.read_table(full_data_path, delimiter=',',
                            names=range(n_f), index_col=False, na_values=['?'])
    df = pd.DataFrame(pd_data,)

    nan_indexes = np.array(df[df.isna().any(axis=1)].index)

    for col_name in df.columns:
        if str(df[col_name].dtype) == 'category' or str(df[col_name].dtype) == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    if len(nan_indexes) != 0:
        df = data_process(df)

    labels_train = df[n_f-1].values
    data_train = df.drop(columns=n_f-1)

    data = data_train.to_numpy()

    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels_train)
    labels[labels != 0] = 1

    print('\nDataset: ', name)
    print('The number of classes: ', len(np.unique(labels, return_counts=True)[0]))
    samples_per_class = [f'class {i}: {samples}' for i, samples in enumerate(np.unique(labels, return_counts=True)[1])]
    print('The number of samples per class: \n', ', '.join(samples_per_class), '\n')
    return data, labels


def data_process(df):
    ''' inpute missing values with means
    Args:
        df: dataframe
    
    Return:
        df: modified dataframe

    '''
    number_cols = df.select_dtypes(include=['float', 'int']).columns
    num_imputer = SimpleImputer(strategy='mean')
    df[number_cols] = num_imputer.fit_transform(df[number_cols])

    return df
