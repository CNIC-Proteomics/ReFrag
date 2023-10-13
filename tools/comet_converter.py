# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:26:09 2023

@author: alaguillog
"""

import pandas as pd

def setMods(instr):
    outstr = instr.split(',')
    for i,j in enumerate(outstr):
        j = j.split('_')
        j = str(j[0]) + str(j[1]) + '(' + str(j[2]) + ')'
        outstr[i] = j
    outstr = ', '.join(outstr)
    return(outstr)

# Files
infile = ''
outfile = ''

# Load
df = pd.read_csv(infile,sep="\t",index_col=False,skiprows=0)

# Change column names
cols = list(df.columns)
cols[0] = 'scannum'
cols[4] = 'precursor_neutral_mass'
cols[12] = 'peptide'
cols[18] = 'modification_info'
cols[21] = 'massdiff'
df.columns = cols

# Format & filter
df = df.dropna(subset='scannum')
df.scannum = df.scannum.astype(int)
# df = df[df.num==1]

# Fixed modifications
df.modification_info = df.apply(lambda x: setMods(x.modification_info), axis=1)

# Write
df.to_csv(outfile, sep="\t", index=False)