import argparse
import os
import pickle
from functools import partial

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

import config
from data.loader import DataLoader
from model.classifier import ClassifyByStyle, sample_weight
from model.cluster import HierarchyCluster
from model.msa import MarketStyleAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='market style analysis')
    parser.add_argument('-s', '--stock', required=True, type=str, help='stock id')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|SenticNet6|LMFinance}')
    parser.add_argument('--ens', type=str, help='market style ensemble classifier {GBDT|AdaBoost}')
    parser.add_argument('--cls', type=str, help='market style cluster {Kmeans|Hierarchy}')
    parser.add_argument('--clf', required=True, type=str, help='sotck price classifier {SVM|GBDT|AdaBoost}')
    return parser.parse_args()


def main():
    args = parse_args()

    # check sentiment lexicon.
    if args.lexicon not in ['SenticNet5', 'SenticNet6', 'LMFinance']:
        raise ValueError('unknown sentiment lexicon.')

    # check classification models' arguments.
    if args.clf == 'SVM':
        clf = partial(SVC, **config.classifier.svc)
    elif args.clf == 'GBDT':
        clf = partial(GradientBoostingClassifier, **config.classifier.gbdt)
    elif args.clf == 'AdaBoost':
        clf = partial(AdaBoostClassifier, **config.classifier.adaboost)
    else:
        raise ValueError('unknown classifier.')
    
    # load classifier parameters.
    name = f'{args.stock}-{args.lexicon}-{args.clf}'
    file  = os.path.join(config.path['clf_param'], f'{name}.pkl')
    with open(file, 'rb') as f:
        clf_params = pickle.load(f)

    # load and process data to be classfied.
    dataset = DataLoader(
        data_dir=config.data.data_dir,
        stock=args.stock,
        lexicon=args.lexicon,
        y_col=config.data.y_col,
        y_offset=config.data.y_offset
    )
    X1, X2 = dataset.get_X(split=config.data.split, norm=config.data.norm)
    Y1, Y2 = dataset.get_y(split=config.data.split, binarize=config.data.binarize)

    # MSA-based classification.
    if args.ens and args.cls:
        # check ensemble models' arguments.
        if args.ens == 'GBDT':
            ens = partial(GradientBoostingClassifier, random_state=config.seed, **config.classifier.gbdt)
        elif args.ens == 'AdaBoost':
            ens = partial(AdaBoostClassifier, random_state=config.seed, **config.classifier.adaboost)
        else:
            raise ValueError('unknown ensemble model.')

        # check cluster models' arguments.
        if args.cls == 'KMeans':
            cls = partial(KMeans, random_state=config.seed, **config.cluster.kmeans)
        elif args.cls == 'Hierarchy':
            cls = partial(HierarchyCluster, random_state=config.seed, **config.cluster.hierarchy)
        else:
            raise ValueError('unknown cluster.')

        # set saving path.
        prefix = f'{args.stock}-{args.lexicon}-{args.clf}-{args.ens}-{args.cls}'
        raw_path   = os.path.join(config.path['test_raw'], f'{prefix}.csv')
        test_path  = os.path.join(config.path['test_metrics'], f'{prefix}.csv')
        style_path = os.path.join(config.path['style'], f'{prefix}.pkl')

        # load msa parameters.
        name = f'{args.stock}-{args.lexicon}-{args.ens}-{args.cls}.pkl'
        file  = os.path.join(config.path['msa_param'], name)
        with open(file, 'rb') as f:
            msa_params = pickle.load(f)
        n, tau = msa_params['n'], msa_params['tau']
        
        # load ensemble parameters.
        name = f'{args.stock}-{args.lexicon}-{args.ens}.pkl'
        file  = os.path.join(config.path['clf_param'], name)
        with open(file, 'rb') as f:
            ens_params = pickle.load(f)
        ens = ens(**ens_params)

        # market style analysis.
        msa = MarketStyleAnalyzer(clf=ens, cls=cls, n_cluster=n, tau=tau, **config.msa.msa)
        train_style=msa.fit_predict(X1)
        test_style=msa.predict(X2)

        # classification data offset.
        X1, X2 = X1.iloc[config.data_offset:], X2.iloc[config.data_offset:]
        Y1, Y2 = Y1.iloc[config.data_offset:], Y2.iloc[config.data_offset:]

        # train and test classification model.
        clf = partial(clf, **clf_params)
        cbs = ClassifyByStyle(clf=clf, n=n, seed=config.seed)
        cbs.fit(X1, Y1, train_style)
        y_pred = cbs.predict(X2, test_style)

        # save styles.
        style = dict(train=train_style, test=test_style)
        with open(style_path, 'wb') as f:
            pickle.dump(style, f)

        # save evaluation results.
        pd.Series(
            dict(
                acc=accuracy_score(Y2, y_pred),
                f1=f1_score(Y2, y_pred),
                recall=recall_score(Y2, y_pred),
                precision=precision_score(Y2, y_pred)
            )
        ).to_csv(test_path, header=False, float_format='%.4f')
        
        # save classification results.
        pd.DataFrame(dict(y_true=Y2, y_pred=y_pred)).to_csv(raw_path, index=False)
    
    # baseline.
    else:
        # set saving path.
        prefix = f'{args.stock}-{args.lexicon}-{args.clf}'
        raw_path  = os.path.join(config.path['baseline_raw'], f'{prefix}.csv')
        test_path = os.path.join(config.path['baseline_metrics'], f'{prefix}.csv')

        # classification data offset.
        X1, X2 = X1.iloc[config.data_offset:], X2.iloc[config.data_offset:]
        Y1, Y2 = Y1.iloc[config.data_offset:], Y2.iloc[config.data_offset:]

        # train and test classification model.
        clf = clf(random_state=config.seed, **clf_params)
        clf.fit(X1, Y1, sample_weight(Y1))
        y_pred = clf.predict(X2)

        # save evaluation results.
        pd.Series(
            dict(
                acc=accuracy_score(Y2, y_pred),
                f1=f1_score(Y2, y_pred),
                recall=recall_score(Y2, y_pred),
                precision=precision_score(Y2, y_pred)
            )
        ).to_csv(test_path, header=False, float_format='%.4f')
        
        # save classification results.
        pd.DataFrame(dict(y_true=Y2, y_pred=y_pred)).to_csv(raw_path, index=False)


if __name__ == '__main__':
    main()