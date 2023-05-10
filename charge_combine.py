# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:03:07 2023

@author: alaguillog
"""

import logging
import pandas as pd
import sys

def main(args):
    dfs = []
    for f in range(0, len(args)):
        dfs += [pd.read_csv(f)]
    dfs = pd.concat(dfs)
    dfs.sort_values(by=['scannum', 'hyperscore'], ascending=[True, False], inplace=True)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='charge_combine',
                                     epilog='Example: python charge_combine.py ch2.txt ch3.txt ch4.txt')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    #if created == 1:
        #logging.info("Created output directory at %s " % args.output)
    main(args)
    logging.info('end script')