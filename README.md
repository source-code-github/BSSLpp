# BSSL++: A Learn++-Inspired Boosting Algorithm for Semi-Supervised Learning

This repository is the official implementation of [BSSL++: A Learn++-Inspired Boosting Algorithm for Semi-Supervised Learning].

## Requirements

This code has been developed under `Python 3.9.16` and `scikit-learn 1.3.0` on `MacOS 13.1 Ventura`.

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

To reproduce the results presented in the paper, run this command:

```
python main.py --data aus --label_size 0.1 --test_size 0.2 --T 10
```

## Data

Datasets are colleclted from UCI (https://archive.ics.uci.edu/) and OpenML(https://openml.org/) dataset repositories.
