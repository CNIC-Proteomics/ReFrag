# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:52 2022

@author: alaguillog
"""

# \\tierra\SC\U_Proteomica\LABS\LAB_FSM\Centrosome_PTMs\RECOM\MSFRAGGER\JAL_NOa2_iTR_FrALL_test\JAL_NOa2_iTR_FrALL.tsv

import argparse
import concurrent.futures
import configparser
import itertools
import logging
import math
import os
import pandas as pd
from pathlib import Path
import pyopenms
import re
import sys
from tqdm import tqdm

def getTquery(fr_ns, mode):
    if mode == "mgf":
        squery = fr_ns.loc[fr_ns[0].str.contains("SCANS=")]
        squery = squery[0].str.replace("SCANS=","")
        squery.reset_index(inplace=True, drop=True)
        mquery = fr_ns.loc[fr_ns[0].str.contains("PEPMASS=")]
        mquery = mquery[0].str.replace("PEPMASS=","")
        mquery.reset_index(inplace=True, drop=True)
        cquery = fr_ns.loc[fr_ns[0].str.contains("CHARGE=")]
        cquery = cquery[0].str.replace("CHARGE=","")
        cquery.reset_index(inplace=True, drop=True)
        tquery = pd.concat([squery.rename('SCANS'),
                            mquery.rename('PEPMASS'),
                            cquery.rename('CHARGE')],
                           axis=1)
        try:
            tquery[['MZ','INT']] = tquery.PEPMASS.str.split(" ",expand=True,)
        except ValueError:
            tquery['MZ'] = tquery.PEPMASS
        tquery['CHARGE'] = tquery.CHARGE.str[:-1]
        tquery = tquery.drop("PEPMASS", axis=1)
        tquery = tquery.apply(pd.to_numeric)
    elif mode == "mzml":
        tquery = []
        for s in fr_ns.getSpectra(): # TODO this is slow
            if s.getMSLevel() == 2:
                df = pd.DataFrame([int(s.getNativeID().split(' ')[-1][5:]), # Scan
                          s.getPrecursors()[0].getCharge(), # Precursor Charge
                          s.getPrecursors()[0].getMZ(), # Precursor MZ
                          s.getPrecursors()[0].getIntensity()]).T # Precursor Intensity
                df.columns = ["SCANS", "CHARGE", "MZ", "INT"]
                tquery.append(df)
        tquery = pd.concat(tquery)
        tquery = tquery.apply(pd.to_numeric)
    return tquery

def readRaw(msdata):
    if os.path.splitext(msdata)[1].lower() == ".mzml":
        mode = "mzml"
        logging.info("\tReading mzML file...")
        fr_ns = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(msdata, fr_ns)
        index2 = 0
        tquery = getTquery(fr_ns, mode)
    elif os.path.splitext(msdata)[1].lower() == ".mgf":
        mode = "mgf"
        logging.info("\tReading MGF file...")
        fr_ns = pd.read_csv(msdata, header=None)
        index2 = fr_ns.to_numpy() == 'END IONS'
        tquery = getTquery(fr_ns, mode)
    else:
        logging.info("MS Data file extension not recognized!")
        sys.exit()
    return(msdata, mode, index2, tquery)

def hyperscore(ions, proof):
    ## 1. Normalize intensity to 10^5
    norm = (ions.INT / ions.INT.max()) * 10E4
    ions["MSF_INT"] = norm
    
    ## 2. Pick matched ions ##
    matched_ions = pd.merge(proof, ions, on="MZ")
    
    ## 3. Adjust intensity
    matched_ions.MSF_INT = matched_ions.MSF_INT / 10E2
    
    ## 4. Hyperscore ##
    matched_ions["SERIES"] = matched_ions.apply(lambda x: x.FRAGS[0], axis=1)
    n_b = matched_ions.SERIES.value_counts()['b']
    n_y = matched_ions.SERIES.value_counts()['y']
    i_b = matched_ions[matched_ions.SERIES=='b'].MSF_INT.sum()
    i_y = matched_ions[matched_ions.SERIES=='y'].MSF_INT.sum()
    
    hs = math.log10(math.factorial(n_b) * math.factorial(n_y) * i_b * i_y)
    return(hs)

def insertMods(peptide, mods):
    mods = mods.split(sep=", ")
    modlist = []
    for m in mods:
        #value = float(re.findall("\d+\.\d+", m)[0])
        pos, value = re.findall('[^()]+', m)
        value = float(value)
        if len(re.findall(r'\d+', pos)) > 0:
            pos = int(re.findall(r'\d+', pos)[0])
        elif pos == "N-term":
            pos = 1
        elif pos == "C-term":
            pos = len(peptide)
        modlist.append([pos, value])
    modlist = modlist[::-1] # Reverse list so it can be added without breaking pos
    for pos, value in modlist:
        peptide = peptide[:pos] + '[' + str(value) + ']' + peptide[pos:] 
    return(peptide)

def parallelFragging(query):
    m_proton = 1.007276
    scan = query.scannum
    charge = query.charge
    MH = query.precursor_neutral_mass + (m_proton*charge)
    plain_peptide = query.peptide
    sequence = insertMods(plain_peptide, query.modification_info)
    retention_time = query.retention_time
    
    return

def main(args):
    '''
    Main function
    '''
    # Parameters
    chunks = float(mass._sections['Parameters']['batch_size'])
    # Read results file from MSFragger
    df = pd.read_csv(Path(args.infile), sep="\t")
    # Read raw file
    msdata, mode, index2, tquery = readRaw(Path(args.rawfile))
    # Prepare to parallelize
    indices, rowSeries = zip(*df.iterrows())
    rowSeries = list(rowSeries)
    tqdm.pandas(position=0, leave=True)
    if len(df) <= chunks:
        chunks = len(df)/args.n_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        refrags = list(tqdm(executor.map(parallelFragging,
                                         rowSeries,
                                         chunksize=chunks),
                            total=len(rowSeries)))
        
    
    
    # Make a Vseq-style query
    # sub = pd.Series([],
    #                 index = ["FirstScan", "Charge", "MH", "Sequence",
    #                          "RetentionTime", "msdataDir", "outDir"])
        
        
    return

if __name__ == '__main__':

    # multiprocessing.freeze_support()
    # parse arguments
    parser = argparse.ArgumentParser(
        description='ReFrag',
        epilog='''
        Example:
            python ReFrag.py

        ''')
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "config/ReFrag.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='MSFragger results file')
    parser.add_argument('-r',  '--rawfile', required=True, help='MS Data file (MGF or MZML)')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.config)
    # if something is changed, write a copy of ini
    if mass.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.infile) + '/ReFrag.ini', 'w') as newconfig:
            mass.write(newconfig)

    # logging debug level. By default, info level
    log_file = outfile = args.infile[:-4] + 'ReFrag_log.txt'
    log_file_debug = outfile = args.infile[:-4] + 'ReFrag_log_debug.txt'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.FileHandler(log_file_debug),
                                      logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')