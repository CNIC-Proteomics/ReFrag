# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:37:42 2023

@author: alaguillog
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import pandas as pd
import sys
pd.options.mode.chained_assignment = None  # default='warn'

def reSort(x):
    return

def main(args):
    
    # Parameters
    indir = args.indir
    rank = args.rank
    scan = args.scan
    score = args.score
    
    # Make results directory
    if not os.path.exists(indir + '\\mzMLjoin'):
        os.mkdir(Path(indir + '\\mzMLjoin'))
    outdir = Path(indir + '\\mzMLjoin')
    
    # Read files
    infiles = []
    for i in glob.glob(indir + '\\*'):
        if (i[-3:].lower()  in ['tsv', 'txt']) and (i[-8:-5].lower() in ['_ch', '_ch', '_ch']): # TODO charges larger than 9 = error
            infiles += [i]
    filenames = list(set([os.path.basename(i)[:-8] for i in infiles])) # TODO charges larger than 9 = error
    
    # Make groups
    for i in filenames:
        df = []
        logging.info('Reading ' + str(i) + '...')
        for k in [j for j in infiles if os.path.basename(j)[:-8] == i]:
            logging.info('\tCharge ' + str(os.path.basename(k)[-5:-4]) + '...') # TODO charges larger than 9 = error
            df += [pd.read_csv(k, sep='\t')]
        df = pd.concat(df)
        # Sort
        df.sort_values([scan, score], ascending=[True, False], inplace=True)
        # Create new rank column
        df.insert(df.columns.get_loc(rank)+1, 'new_hit_rank', df.groupby(scan).cumcount()+1)
        # Write output
        logging.info('Writing ' + str(i) + '.tsv...')
        df.to_csv(os.path.join(outdir, i + '.tsv'), sep='\t', index=False)
    logging.info('Done.')
    return

if __name__ == '__main__':

    # multiprocessing.freeze_support()
    # parse arguments
    parser = argparse.ArgumentParser(
        description='mzMLjoin',
        epilog='''
        Example:
            python mzMLjoin.py

        ''')
    
    parser.add_argument('-i',  '--indir', required=True, help='Input directory')
    parser.add_argument('-s',  '--scan', default='scannum', type=str, help='Scan column name (default: %(default)s)')
    parser.add_argument('-r',  '--rank', default='hit_rank', type=str, help='Rank column name (default: %(default)s)')
    parser.add_argument('-x',  '--score', default='hyperscore', type=str, help='Score column name (default: %(default)s)')
    parser.add_argument('-w',  '--n_workers', type=int, default=os.cpu_count(), help='Number of threads/n_workers')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')