import argparse
import os
import pickle
import sys

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

sys.path.append('.')
import config
from _typing import ArrayLike
from data.loader import DataLoader
from model.classifier import sample_weight


def parse_args():
    parser = argparse.ArgumentParser(description='market style analysis')
    parser.add_argument('-s', '--stock', required=True, type=str, help='stock id')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|SenticNet6|LMFinance}')
    parser.add_argument('--clf', required=True, type=str, help='sotck price classifier {SVM|GBDT|AdaBoost}')
    parser.add_argument('-v', '--verbose', action='store_true', help='output verbose')
    return parser.parse_args()


def scoring(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    r   = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    acc = balanced_accuracy_score(y_true, y_pred)
    return 0 if r in [0, 1] else min(f1, acc)


def main():
    args = parse_args()

    # check classification models' arguments.
    if args.clf == 'SVM':
        clf  = SVC(**config.classifier.svc, random_state=config.seed)
        grid = config.classifier.grids.svc
    elif args.clf == 'GBDT':
        clf = GradientBoostingClassifier(**config.classifier.gbdt, random_state=config.seed)
        grid = config.classifier.grids.gbdt
    elif args.clf == 'AdaBoost':
        grid = config.classifier.grids.adaboost
        clf = AdaBoostClassifier(**config.classifier.adaboost, random_state=config.seed)
    else:
        raise ValueError('unknown classifier.')

    # set file name.
    prefix = f'{args.stock}-{args.lexicon}-{args.clf}'

    # load dataset.
    dataset = DataLoader( 
        data_dir=config.data.data_dir,
        stock=args.stock,
        lexicon=args.lexicon,
        y_col=config.data.y_col,
        y_offset=config.data.y_offset
    )
    X1, _ = dataset.get_X(split=config.data.split, norm=config.data.norm)
    Y1, _ = dataset.get_y(split=config.data.split, binarize=config.data.binarize)
    X1 = X1.iloc[config.data_offset:]
    Y1 = Y1.iloc[config.data_offset:]

    # grid search.
    score    = make_scorer(scoring, greater_is_better=True)
    searcher = GridSearchCV(clf, grid, scoring=score, **config.classifier.grid_search)
    searcher.fit(X1, Y1, sample_weight=sample_weight(Y1))
    
    # save results.
    save_path = os.path.join(config.path['clf_gs'], f'{prefix}.csv')
    pd.DataFrame(searcher.cv_results_).to_csv(save_path, index=False, float_format='%.4f')
    save_path = os.path.join(config.path['clf_param'], f'{prefix}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(searcher.best_params_, f)
    
    # output results.
    if args.verbose:
        y_pred = searcher.predict(X1)
        f1  = f1_score(Y1, y_pred)
        r   = recall_score(Y1, y_pred)
        p   = precision_score(Y1, y_pred)
        acc = balanced_accuracy_score(Y1, y_pred)
        print(f'best score: {searcher.best_score_:.5f}.')
        print(f'best param: {searcher.best_params_}.')
        print(f'f1: {f1:.5f}, recall: {r:.5f}, precision: {p:.5f}, balanced acc:{acc:.5f}.')

    

if __name__ == '__main__':
    main()