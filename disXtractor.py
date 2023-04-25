# -*- coding: utf-8 -*-
# Created on Mon Oct 29 15:56:36 2018
# Module metadata variables
__author__ = ["Andrea Laguillo Gómez", "Ricardo Magni"]
__credits__ = ["Andrea Laguillo Gómez", "Ricardo Magni", "Jesus Vazquez"]
__license__ = "Creative Commons Attribution-NonCommercial-NoDerivs 4.0 Unported License https://creativecommons.org/licenses/by-nc-nd/4.0/"
__version__ = "0.1.0"
__maintainer__ = "Andrea Laguillo Gómez"
__email__ = "andrea.laguillo@cnic.es"
__status__ = "Development"

import argparse
import dask.dataframe as dd
from dask.delayed import delayed
import glob
import logging
import numpy as np
import os
import pandas as pd
import sys

def targetdecoy(df,decoy_prefix):
    '''Label target and decoy identifications'''
    z=list(df["protein"].str.split(";"))
    p=[(all(decoy_prefix  in item for item in i )) for i in z]
    p=list(map(int, p))
    return p

def protein(df):
    proteins=[list(i) for i in list(zip(df.protein,df["Label"]))]
    for p in proteins:
        if p[1]==1:
            p[0]=p[0].split(",")[0]
        else:
            p[0]=[s for s in p[0].split(",") if "sp" or "tr" in s]                       
            p[0].sort()
            p[0]=p[0][0]
    proteins=[x[0] for x in proteins]
    return proteins

def DiSXtractor(file, root, fdrthreshold, Decoy_tag, score, peptide, calc_cxcorr, skip):
        logging.info("Reading input file...")
        if os.path.splitext(file)[1].lower() == "feather":
            df = pd.read_feather(file)
        else:
            df = pd.read_csv(file,sep="\t",index_col=False,skiprows=skip)
        #df["File"]=os.path.basename(file)
        logging.info("Preprocessing...")
        if 'Label' not in df:
            df["Label"]=targetdecoy(df,Decoy_tag)
        else:
            df["Type"] = df['Label']
            df["Label"] = df.apply(lambda x: 1 if x['Label']=="Decoy" else 0, axis = 1)
        #df["protein"]=protein(df)
        if 'Description' in df:
            df["Gene"]=df.Description.str.split("|").str[-1].str.split("_").str[0]
        else:
            df["Gene"]=df.protein.str.split("|").str[2].str.split(",").str[0].str.split("_").str[0]
        if calc_cxcorr == 1:
            if 'Charge' in df: # MSFragger, PD
                df["R"]=np.where(df['Charge']>=3, '1.22', '1').astype(float)
                df["cXcorr"]= np.log(df[score]/df.R)/np.log(2*df.Sequence.str.len())
            else: # Comet, Recom
                df["R"]=np.where(df['charge']>=3, '1.22', '1').astype(float)
                df["cXcorr"]= np.log(df[score]/df.R)/np.log(2*df.peptide.str.len())
            df=df.sort_values(by="cXcorr", ascending=False)
        else:
            df=df.sort_values(by=score, ascending=False)
        s=df.shape[0]
                    
        "*********** Calculate FDR and probability for hyperscore ************"
        logging.info("Calculating FDR and probability for " + str(score) + "...")
        if calc_cxcorr == 1:
            df=df.sort_values(by="cXcorr", ascending=False)
        else:
            df=df.sort_values(by=score, ascending=False)
        df["rank"]=df.groupby("Label").cumcount()+1
        df["rank_T"]=np.where(df["Label"]==0,df["rank"],0)
        df["rank_T"]=df["rank_T"].replace(to_replace=0, method='ffill')
        df["rank_D"]=np.where(df["Label"]==1,df["rank"],0)
        df["rank_D"]=df["rank_D"].replace(to_replace=0, method='ffill')
        df["FDR_H"]=df["rank_D"]/df["rank_T"]
        df["P_H"] = df["rank_D"]/s
        
        "*********** Calculate FDR and probability for sp_score **************"
        logging.info("Calculating FDR and probability for sp_score...")
        df=df.sort_values(by="REFRAG_sp_score", ascending=False)
        df["rank"]=df.groupby("Label").cumcount()+1
        df["rank_T"]=np.where(df["Label"]==0,df["rank"],0)
        df["rank_T"]=df["rank_T"].replace(to_replace=0, method='ffill')
        df["rank_D"]=np.where(df["Label"]==1,df["rank"],0)
        df["rank_D"]=df["rank_D"].replace(to_replace=0, method='ffill')
        df["FDR_SP"]=df["rank_D"]/df["rank_T"]
        df["P_SP"] = df["rank_D"]/s
        
        "************** Calculate FDR and probability for CP *****************"
        logging.info("Calculating combined FDR and probability...")
        df["CP"]= df["P_H"]*0.5 + df["P_SP"]*0.5 
        df=df.sort_values(by="CP", ascending=True)
        df["rank"]=df.groupby("Label").cumcount()+1
        df["rank_T"]=np.where(df["Label"]==0,df["rank"],0)
        df["rank_T"]=df["rank_T"].replace(to_replace=0, method='ffill')
        df["rank_D"]=np.where(df["Label"]==1,df["rank"],0)
        df["rank_D"]=df["rank_D"].replace(to_replace=0, method='ffill')
        df["FDR_CP"]=df["rank_D"]/df["rank_T"]
        df=df[df["FDR_CP"] <= fdrthreshold]
        df=df[df["Label"]==0]
        logging.info("Writing output file...")
        if os.path.isdir(root):
            df.to_csv(os.path.join(root,
                                   os.path.basename(file).replace(".tsv",
                                                                  "_FDR_"+str(fdrthreshold)+".tsv")),
                      sep="\t",
                      index=False)
        else:
            df.to_csv(file.replace("."+os.path.basename(file).split(".")[1],
                                   "_FDR_"+str(fdrthreshold)+".tsv"),
                      sep="\t",
                      index=False)
        return df

def main(args):
    root = args.input
    fdrthreshold = args.fdr
    Decoy_tag = args.decoy
    score = args.score
    peptide = args.peptide
    calc_cxcorr = args.correct
    skip = args.skip
    logging.info("Reading files...")
    if os.path.isdir(root):
        listfiles = (glob.glob(os.path.join(root,"*.tsv"),recursive=True) +
                     glob.glob(os.path.join(root,"*.txt"),recursive=True) +
                     glob.glob(os.path.join(root,"*.feather"),recursive=True))
    else:
        listfiles = [root]
    ddf = dd.from_delayed([delayed(DiSXtractor) (file, root, fdrthreshold, Decoy_tag, score, peptide, calc_cxcorr, skip) for file in listfiles]).compute(scheduler="processes")
    if os.path.isdir(root):
        logging.info("Writing combined output file...")
        ddf.to_csv(os.path.join(root,"All_FDR_"+str(fdrthreshold)+".tsv"),sep="\t",index=False)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Peak FDRer',
                                     epilog='Example: python PeakFDRer.py')
    parser.add_argument('-i',  '--input', required=True, help='Path to input directory')
    parser.add_argument('-f',  '--fdr', type=float, default=0.01, help='FDR threshold (default: %(default)s)')
    parser.add_argument('-d',  '--decoy', type=str, default='DECOY', help='Decoy tag (default: %(default)s)')
    parser.add_argument('-s',  '--score', default='hyperscore', help='Name of score column (default: %(default)s)')
    parser.add_argument('-p',  '--peptide', default='peptide', help='Name of peptide column (default: %(default)s)')
    parser.add_argument('-k',  '--skip', type=int, default=0, help='Skip first N lines (default: %(default)s)')
    parser.add_argument('-c',  '--correct', type=bool, default=0, help='Calculate corrected score, 0=NO 1=YES (default: %(default)s)')
    args = parser.parse_args()
    # Set up logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=[logging.StreamHandler()])
    # Start
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')

