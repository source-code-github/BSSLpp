import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import ConvergenceWarning
from pandas.errors import SettingWithCopyWarning

from sklearn.neural_network import MLPClassifier
from utils.bsslpp import BSSLpp
from utils.datasets import get_data

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='pid', 
                    choices=['aus', 'hms', 'hdc', 'hep', 'kvk', 'mm','pid', 'vote', 'wdbc'],
                    help='name of the dataset')
parser.add_argument('--label_size', type=float,
                    default=0.2, help='labeled set proportion ranging from 0.1 to 1.0')
parser.add_argument('--test_size', type=float,
                    default=0.2, help='test set proportion rangin from 0.1 to 1.0')
parser.add_argument('--T', type=int,
                    default=10, help='number of boosting iterations')
parser.add_argument('--SS', type=int,
                    default=20, help='number of random shuffle and splits')


opt = parser.parse_args()

def main():
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

    print('hello')

    test_accs = []

    dataset_name = opt.dataset
    data, labels = get_data(dataset_name)

    n_classes = len(np.unique(labels))

    n_splits = opt.SS

    Tk = opt.T
    test_size = opt.test_size
    label_size = opt.label_size

    # MLP parameters
    solver = 'adam'
    activation = 'logistic'
    batch_size = 32
    hidden_layer = (data.shape[1] * 2, ) # single hidden layer with 2*p neuros, p is the number of features

    count_splits = 0
    print(f'Running ...')

    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42)
    for train_unl_index, test_index in sss.split(data, labels):
        X_unl_train, X_test = data[train_unl_index], data[test_index]
        y_unl_train, y_test = labels[train_unl_index], labels[test_index]

        X_lab, X_unl_all, y_lab, y_unl_all = train_test_split(
            X_unl_train, y_unl_train, train_size=label_size, random_state=42, stratify=y_unl_train)


        count_splits += 1
        
        scaler = StandardScaler()
        scaler.fit(X_lab)
        X_lab = scaler.transform(X_lab)
        X_unl_all = scaler.transform(X_unl_all)
        X_test = scaler.transform(X_test)

        model_lpp_ssl = BSSLpp(
            n_classes=n_classes, Tk=Tk, len_lab=len(y_lab),
            base_estimator=MLPClassifier(
                random_state=42, batch_size=batch_size, activation=activation, solver=solver,
                hidden_layer_sizes=hidden_layer, max_iter=300))

        model_lpp_ssl.fit(X_lab, y_lab, X_unl_all)

        test_acc = model_lpp_ssl.score(X_test, y_test)

        test_accs.append(test_acc)

        print(f'splits completed: {count_splits} / {n_splits}')

    print('Result: mean err: {:.1f}, and std: {:.1f}'.format( (1-np.mean(test_accs))*100, np.std(test_accs)*100) ) 


if __name__ == '__main__':
    main()
