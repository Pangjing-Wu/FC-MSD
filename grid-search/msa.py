import argparse
import os
import pickle
import sys
from functools import partial

from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

sys.path.append('.')
import config
from data.loader import DataLoader
from model.cluster import HierarchyCluster
from model.msa import GridSearchMSA, MarketStyleAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='market style analysis')
    parser.add_argument('-s', '--stock', required=True, type=str, help='stock id')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|SenticNet6|LMFinance}')
    parser.add_argument('--ens', type=str, help='market style ensemble classifier {GBDT|AdaBoost}')
    parser.add_argument('--cls', type=str, help='market style cluster {KMeans|Hierarchy}')
    parser.add_argument('-v', '--verbose', action='store_true', help='output verbose')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # check sentiment lexicon.
    if args.lexicon not in ['SenticNet5', 'SenticNet6', 'LMFinance']:
        raise ValueError('unknown sentiment lexicon.')

    # check ensemble models' arguments.
    if args.ens == 'GBDT':
        ens = partial(GradientBoostingClassifier, **config.classifier.gbdt)
    elif args.ens == 'AdaBoost':
        ens = partial(AdaBoostClassifier, **config.classifier.adaboost)
    else:
        raise ValueError('unknown ensemble model.')

    # check cluster models' arguments.
    if args.cls == 'KMeans':
        cls    = partial(KMeans, **config.cluster.kmeans)
        repeat = config.msa.search_repeat
    elif args.cls == 'Hierarchy':
        cls    = partial(HierarchyCluster, **config.cluster.hierarchy)
        repeat = 1
    else:
        raise ValueError('unknown cluster.')

    # set file name.
    prefix = f'{args.stock}-{args.lexicon}-{args.ens}-{args.cls}'


    # load dataset
    dataset = DataLoader(
        data_dir=config.data.data_dir,
        stock=args.stock,
        lexicon=args.lexicon,
        y_col=config.data.y_col,
        y_offset=config.data.y_offset
    )
    X1, _ = dataset.get_X(split=config.data.split, norm=config.data.norm)

    # set models.
    msa = partial(MarketStyleAnalyzer, **config.msa.msa)
    searcher = GridSearchMSA(msa=msa, clf=ens, cls=cls, **config.msa.grid_search)

    # grid search.
    msa = searcher.fit(X1, repeat, config.seed)

    # save grid search results.
    save_path = os.path.join(config.path['msa_gs'], f'{prefix}.csv')
    searcher.search_results.to_csv(save_path, index=False, float_format='%.4f')

    # save market style analysis results.
    save_path = os.path.join(config.path['msa_param'], f'{prefix}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(searcher.best_parameters, f)

    if args.verbose:
        print(f'best score: {searcher.best_score:.5f}.')
        print(f'best param: {searcher.best_parameters}.')


if __name__ == '__main__':
    main()