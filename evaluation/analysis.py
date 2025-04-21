import matplotlib.pyplot as plt
import pickle
from os.path import dirname, join
from functools import reduce
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier

from datasets import load


def get_common_features(pickle_names, probe_ref_dataset, verbose=True):
    folder = join(dirname(__file__), '..', 'datasets')

    # Load all selected feature names from pickle files
    genes = []
    for ds in pickle_names:
        with open(join(folder, ds, f"{ds}.pickle"), 'rb') as f:
            genes.append(pickle.load(f))

    common = reduce(np.intersect1d, genes)

    if verbose:
        print(f"\ {len(common)} common features selected across: {', '.join(pickle_names)}\n{common}\n")

    # Map back to index from the reference dataset (assume all share probes)
    with open(join(folder, probe_ref_dataset, "common_probes.txt"), "r") as f:
        probes = [p for p in f.read().splitlines() if p]
    g_idx = [probes.index(gene) for gene in common]

    return common, np.array(g_idx)


def evaluate_on_all_datasets(common_idx, datasets, clf=None, cv=10):
    clf = clf or ExtraTreesClassifier(n_estimators=20, random_state=0)

    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'recall_macro': 'recall_macro'
    }

    print("Dataset\t\tTrainAcc\tTestAcc\t\tTrainPrec\tTestPrec\tTrainRecall\tTestRecall")
    for name in datasets:
        X, y, _, _, _ = load._loads[name]()
        X = X[:, common_idx]
        scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=True)

        print(f"{name:<10}\t"
              f"{scores['train_acc'].mean():.2f}\t\t{scores['test_acc'].mean():.2f}\t\t"
              f"{scores['train_prec_macro'].mean():.2f}\t\t{scores['test_prec_macro'].mean():.2f}\t\t"
              f"{scores['train_recall_macro'].mean():.2f}\t\t{scores['test_recall_macro'].mean():.2f}")


def unpickle_results():
    datasets = ['ad_kronos', 'ad_rush', 'ad_blood1', 'ad_blood2']
    common_features, common_indices = get_common_features(datasets, probe_ref_dataset='ad_rush')
    evaluate_on_all_datasets(common_indices, datasets)


if __name__ == "__main__":
    unpickle_results()
