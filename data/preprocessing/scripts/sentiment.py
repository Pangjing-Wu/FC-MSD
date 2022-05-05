import argparse
import glob
import os
import sys

import pandas as pd

sys.path.append('.')
from data.preprocessing import *


def parse_args():
    parser = argparse.ArgumentParser(description= 'batch calculate news sentiments')
    parser.add_argument('-i', '--data_dir', required=True, type=str, help='direction of data file')
    parser.add_argument('-o', '--save_dir', required=True, type=str, help='direction of output file')
    parser.add_argument('--lexicon', required=True, type=str, help='sentiment lexicion {SenticNet5|SenticNet6|LMFinance}')
    parser.add_argument('--lexicon_dir', type=str, help='lexicion file direction')
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.data_dir):
        csvlist = glob.glob(os.path.join(args.data_dir, '*.csv'))
    elif os.path.isfile(args.data_dir):
        csvlist = [args.data_dir]
    else:
        raise KeyError('unknown data direction.')

    if args.lexicon == 'SenticNet5':
        lexicon = SenticNet5()
    elif args.lexicon == 'SenticNet6':
        lexicon = SenticNet6()
    elif args.lexicon == 'LMFinance':
        lexicon = LMFinance(args.lexicon_dir)
    else:
        raise KeyError('unknown sentiment lexicon.')

    savedir = os.path.join(args.save_dir, args.lexicon)
    os.makedirs(savedir, exist_ok=True)
    print(f"load {len(csvlist)} file(s) from '{args.data_dir}', save to '{savedir}'.")

    for i, csvfile in enumerate(csvlist):
        news = pd.read_csv(csvfile)
        data = pd.DataFrame(columns=['date', *lexicon.sentiment_tags], index=news.index, dtype='float')
        data.date = news.date
        for index, content in zip(news.index, news.contents):
            score = lexicon.score(content)
            for sentiment in score:
                data.loc[index, sentiment] = score[sentiment] 
        data = data.dropna(axis=0)
        data = data.groupby('date').agg('sum')
        filename = os.path.basename(csvfile)
        data.to_csv(os.path.join(savedir, filename), float_format='%.5f')
        print(f'[{i+1:2}/{len(csvlist):2}] {filename} was preprocessed.')
    print('All files have been preprocessed.')


# python -u ./scripts/sentiment.py -i ./dataset/raw/news -o ./dataset/processed/sentiments --lexicon SenticNet6
if __name__ == "__main__":
    main()
