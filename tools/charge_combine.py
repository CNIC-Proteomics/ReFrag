# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:03:07 2023

@author: alaguillog
"""

import argparse
import glob
import logging
import os
import pandas as pd
import sys

def main(args):
    # Get filenames
    all_infiles = []
    all_filenames = []
    for a in range(0, len(args)):
        f = args[a]
        infiles = glob.glob(f + '/*.tsv')
        all_infiles.append(infiles)
        filenames = list(set([os.path.basename(i)[:-4] for i in infiles]))
        all_filenames.append(filenames)
    if not os.path.exists(os.path.dirname(all_infiles[0][0])+r'/ALL_CHARGES'):
        os.mkdir(os.path.dirname(all_infiles[0][0])+r'/ALL_CHARGES')
    outdir = os.path.dirname(all_infiles[0][0])+r'/ALL_CHARGES/'
    # Check
    ref = set(all_filenames[0])
    for i in all_filenames:
        if set(i) != ref:
            sys.exit('ERROR: Missing files!')
    logging.info(str(len(ref)) + ' files to join.')
    # Join
    all_infiles = [item for sublist in all_infiles for item in sublist]
    for i in sorted(ref):
        logging.info('Joining ' + str(i) + '...')
        dfs = []
        subset = [f for f in all_infiles if os.path.basename(f)[:-4]==i]
        for s in subset:
            dfs += [pd.read_csv(s, sep='\t')]
        dfs = pd.concat(dfs)
        dfs.sort_values(by=['scannum', 'hyperscore'], ascending=[True, False], inplace=True)
        dfs = dfs.groupby('scannum').head(5)
        # Write
        dfs.to_csv(outdir+str(i)+'.tsv', sep='\t', index=False)

if __name__ == '__main__':
    # parse arguments
    # parser = argparse.ArgumentParser(description='charge_combine',
    #                                  epilog='Example: python charge_combine.py path/ch2/ path/ch3/ path/ch4/')
    # args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        handlers=[logging.StreamHandler()])
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    #if created == 1:
        #logging.info("Created output directory at %s " % args.output)
    if len(sys.argv) <= 1:
        sys.exit('USAGE: python charge_combine.py path/ch2/ path/ch3/ path/ch4/')
    else:
        main(sys.argv[1:])
    logging.info('end script')