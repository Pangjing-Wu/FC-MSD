import argparse
import glob
import os
import sys

import pandas as pd
sys.path.append('.')
from data.preprocessing import CalculateIndicators


def parse_args():
  parser = argparse.ArgumentParser(description= 'calculate technical indicators')
  parser.add_argument('-i', '--data_dir', required=True, type=str, help='direction of data file')
  parser.add_argument('-o', '--save_dir', required=True, type=str, help='direction of output file')
  return parser.parse_args()


def main():
    params = parse_args()

    if os.path.isdir(params.data_dir):
        csvlist = glob.glob(os.path.join(params.data_dir, '*.csv'))
    elif os.path.isfile(params.data_dir):
        csvlist = [params.data_dir]
    else:
        raise KeyError('unknown data direction')

    os.makedirs(params.save_dir, exist_ok=True)

    print('load data from %s, save to %s.' % (params.data_dir, params.save_dir))

    for i, csvfile in enumerate(csvlist):
        indicator = CalculateIndicators()
        price = pd.read_csv(csvfile)
        data  = indicator.get_indicator(price)
        data  = data.dropna(axis=0)
        filename = os.path.basename(csvfile)
        data.to_csv(os.path.join(params.save_dir, filename), index=False, float_format='%.5f')
        print(f'[{i+1:2}/{len(csvlist):2}] {filename} was preprocessed.')
        
    print('All files have been preprocessed.')


if __name__ == '__main__':
    main()