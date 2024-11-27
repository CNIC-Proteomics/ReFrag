# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:52 2022

@author: alaguillog
"""

from ast import literal_eval
import argparse
from collections import defaultdict
import concurrent.futures
import configparser
from datetime import datetime
import itertools
import logging
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyopenms
import re
import scipy.stats
import statistics
import string
import sys
import shutup
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'
shutup.please()

def checkParams(mass):
    min_frag_mz = int(mass._sections['Spectrum Processing']['min_fragment_mz'])
    max_frag_mz = int(mass._sections['Spectrum Processing']['max_fragment_mz'])
    if (max_frag_mz > 0) & (max_frag_mz <= min_frag_mz):
        logging.error('max_frag_mz must be either 0 or a value greater than min_frag_mz')
        return(1)
    return(0)

def preProcess(args):
    if os.path.isdir(args.infile):
        infiles = []
        for f in os.listdir(args.infile):
            if f.lower().endswith(".tsv"):
                if len(args.dia) > 0:
                    infiles += [os.path.join(args.infile, os.path.basename(f).split(sep=".")[0] + "_ch" + str(c) + ".tsv") for c in args.dia]
                else:
                    infiles += [os.path.join(args.infile, f)]
        if len(infiles) == 0:
            sys.exit("ERROR: No TSV files found in directory " + str(args.infile))
    else:
        if len(args.dia) > 0:
            infiles = [os.path.join(os.path.dirname(args.infile), os.path.basename(args.infile).split(sep=".")[0] + "_ch" + str(c) + ".tsv") for c in args.dia]
        else:
            infiles = [args.infile]

    if os.path.isdir(args.rawfile):
        rawfiles = []
        rawbase = []
        for f in os.listdir(args.rawfile):
            if (f.lower().endswith(".mzml") or f.lower().endswith(".mgf")):
                rawfiles += [os.path.join(args.rawfile, f)]
                rawbase += [os.path.basename(f).split(sep=".")[0]]
        if len(rawfiles) == 0:
            sys.exit("ERROR: No mzML or MGF files found matching pattern " + str(args.rawfile))
    else:
        rawfiles = [args.rawfile]
        rawbase = [os.path.basename(args.rawfile).split(sep=".")[0]]

    return(infiles, rawfiles, rawbase)

def readRaw(msdata):
    if os.path.splitext(msdata)[1].lower() == ".mzml":
        mode = "mzml"
        logging.info("Reading mzML file (" + str(os.path.basename(Path(msdata))) + ")...")
        fr_ns = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(str(msdata), fr_ns)
        index2 = 0
        logging.info("\t" + str(fr_ns.getNrSpectra()) + " spectra read.")
    elif os.path.splitext(msdata)[1].lower() == ".mgf":
        mode = "mgf"
        logging.info("Reading MGF file (" + str(os.path.basename(Path(msdata))) + ")...")
        fr_ns = pd.read_csv(msdata, header=None)
        index2 = fr_ns.to_numpy() == 'END IONS'
        logging.info("\t" + str(sum(fr_ns[0].str[:4]=="SCAN")) + " spectra read.")
        # logging.info("\t" + str(fr_ns[0].str.count('SCANS').sum()) + " spectra read.") # VERY SLOW
    else:
        logging.info("MS Data file type '" + str(os.path.splitext(msdata)[1]) + "' not recognized!")
        sys.exit()
    return(fr_ns, mode, index2)

def deisotope(ions, m_proton, max_charge):
    # Sort by descending intensity
    ions = np.array([ions[0][ions[1].argsort()][::-1], ions[1][ions[1].argsort()][::-1]])
    if len(ions[0]) > 1000: # Remove low intensity peaks
        ions = np.array([ions[0][:999], ions[1][:999]])
    # Sort by ascending m/z
    ions = np.array([ions[0][ions[0].argsort()], ions[1][ions[0].argsort()]])
    isoids = []
    for i in range(1, max_charge): # Consider charges up to precursor charge state - 1
        isoids1 = [np.isclose(ions[0], j-(m_proton/i), atol=0.005, rtol=0).any() for j in ions[0]]
        isoids2 = [np.isclose(ions[0], j-((m_proton*2)/i), atol=0.005, rtol=0).any() for j in ions[0]]
        isoids3 = [np.isclose(ions[0], j-((m_proton*2)/i), atol=0.005, rtol=0).any() for j in ions[0]]
        isoids += [np.logical_or.reduce((isoids1, isoids2, isoids3))]
    isoids = np.logical_or.reduce(isoids)
    ions = np.array([np.delete(ions[0], isoids), np.delete(ions[1], isoids)])
    return(ions)

def locateScan(scan, mode, fr_ns, spectra, spectra_n, index2, top_n, bin_top_n, min_ratio,
               min_frag_mz, max_frag_mz, m_proton, deiso, max_charge):
    if mode == "mgf":
        # index1 = fr_ns.to_numpy() == 'SCANS='+str(int(scan))
        try:
            index1 = fr_ns.loc[fr_ns[0]=='SCANS='+str(scan)].index[0] + 1
            # index1 = np.where(index1)[0][0]
        except IndexError:
            logging.info("\tERROR: Scan number " + str(scan) + " not found in MGF file.")
            sys.exit()
        index3 = np.where(index2)[0]
        index3 = index3[np.searchsorted(index3,[index1,],side='right')[0]]
        try:
            ions = fr_ns.iloc[index1:index3,:]
            ions[0] = ions[0].str.strip()
            ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
            ions = ions.drop(ions.columns[0], axis=1)
            ions = ions.apply(pd.to_numeric)
        except ValueError:
            ions = fr_ns.iloc[index1+4:index3,:]
            ions[0] = ions[0].str.strip()
            ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
            ions = ions.drop(ions.columns[0], axis=1)
            ions = ions.apply(pd.to_numeric)
        ions = np.array(ions.T)
    elif mode == "mzml":
        try:
            s = spectra[spectra_n.index(scan)]
        except AssertionError or OverflowError:
            logging.info("\tERROR: Scan number " + str(scan) + " not found in mzML file.")
            sys.exit()
        peaks = s.get_peaks()
        ions = np.array([peaks[0], peaks[1]])
    # Deisotope (experimental)
    if deiso: # TODO: Intensity, 2C13
        ions = deisotope(ions, m_proton, max_charge)
    # Remove peaks below min_ratio
    cutoff = np.where(ions[1]/max(ions[1]) >= min_ratio)
    ions = np.array([ions[0][cutoff], ions[1][cutoff]])
    # Remove peaks outside range
    if max_frag_mz > 0:
        cutoff = np.where((ions[0] >= min_frag_mz) & (ions[0] <= max_frag_mz))
    else:
        cutoff = np.where(ions[0] >= min_frag_mz)
    ions = np.array([ions[0][cutoff], ions[1][cutoff]])
    # Return only top N peaks
    if bin_top_n:
        bins = np.digitize(ions[0], np.arange(55,max(ions[0]),110))
        ions_f = []
        for i in np.unique(bins):
            ions_t = np.array([ions[0][np.where(bins==i)], ions[1][np.where(bins==i)]])
            cutoff = len(ions_t[0])-top_n
            if (cutoff < 0) or (cutoff >= len(ions_t[0])): cutoff = 0
            ions_f += [np.array([ions_t[0][ions_t[1].argsort()][cutoff:], ions_t[1][ions_t[1].argsort()][cutoff:]])]
        ions = np.concatenate(ions_f, axis=1)
    else:
        cutoff = len(ions[0])-top_n
        if (cutoff < 0) or (cutoff >= len(ions[0])): cutoff = 0
        ions = np.array([ions[0][ions[1].argsort()][cutoff:], ions[1][ions[1].argsort()][cutoff:]])
    if len(np.unique(ions[0])) != len(ions[0]): # Duplicate m/z measurement
        ions = pd.DataFrame(ions).T
        ions = ions[ions.groupby(0)[1].rank(ascending=False)<2]
        ions.drop_duplicates(subset=0, inplace=True)
        ions = np.array(ions.T)
    return(ions)

def hyperscore(ch, ions, proof, pfrags, ftol=50): # TODO play with number of ions # if modified frag present, don't consider non-modified?
    # def _stirling(x):
    #     y = x * math.log(x) - x + 0.5 * math.log(x) + 0.5 * math.log(math.pi * 2.0 * x)
    #     return
    ## 1. Normalize intensity
    MSF_INT = (ions[1] / ions[1].max()) * 10E2
    ## 2. Pick matched ions ##
    pfrags = pfrags[proof[1]<=ftol]
    proof = np.array([proof[0][proof[1]<=ftol],
                      proof[1][proof[1]<=ftol],
                      proof[2][proof[1]<=ftol]])
    matched_ions = np.array([proof[0], proof[1], proof[2], 
                             np.repeat(MSF_INT[np.isin(ions[0], proof[0])], np.unique(proof[0], return_counts=True)[1])])
    if (len(matched_ions[0]) == 0) or (len(pfrags) == 0):
        hs = 0
        return(hs, 0)
    ## Deconvolute charges ##
    SERIES = [''.join(filter(str.isalnum, i)) for i in pfrags]
    SERIES = np.transpose(np.array([SERIES, matched_ions[3]]))
    lookup = defaultdict(lambda: 0)
    for f, i in SERIES:
        lookup[f] = float(lookup[f]) + float(i)
    SERIES = np.transpose([[f, i] for f, i in lookup.items()])
    INTENS = np.array([float(i) for i in SERIES[1]])
    SERIES = SERIES[0].astype('<U1')
    ## 3. Hyperscore ##
    # SERIES = pfrags.astype('<U1')
    #SERIES_C = (np.unique(np.array([f.replace('+' , '') for f in pfrags]))).astype('<U1') # Group fragment charges
    if len(INTENS[SERIES == 'b']) == 0:
        n_b = 1 # So that hyperscore will not be 0 if one series is missing
        i_b = 1
    else:
        n_b = (SERIES == 'b').sum()
        i_b = INTENS[SERIES == 'b'].sum()
    if len(INTENS[SERIES == 'y']) == 0:
        n_y = 1 # So that hyperscore will not be 0 if one series is missing
        i_y = 1
    else:
        n_y = (SERIES == 'y').sum()
        i_y = INTENS[SERIES == 'y'].sum()
    try:
        #hs = math.log(math.factorial(n_b) * math.factorial(n_y)) + math.log(i_b * i_y)
        hs = math.log((i_b + 1) * (i_y + 1)) + math.log(math.factorial((n_b))) + math.log(math.factorial(n_y))
    except ValueError:
        hs = 0
    if hs < 0:
        hs = 0
    # TODO: CHARGE CORRECTION
    # TODO: sequence length correction?
    return(hs, i_b+i_y)

def spscore(sub_spec, matched_ions, ftol, seq, mfrags):
    if mfrags.size > 0:
        # Consecutive fragments
        f = np.unique(mfrags)
        # B series
        bf = f[np.char.startswith(f, 'b')]
        if bf.size > 0:
            bf = np.array([i.replace('b' , '') for i in bf])
            bf3 = bf[np.char.endswith(bf, '+++')]
            bf = bf[np.invert(np.char.endswith(bf, '+++'))]
            bf2 = bf[np.char.endswith(bf, '++')]
            bf = bf[np.invert(np.char.endswith(bf, '++'))]
            bf3 = np.array([int(i.replace('+' , '')) for i in bf3])
            bf2 = np.array([int(i.replace('+' , '')) for i in bf2])
            bf = np.array([int(i) for i in bf])
        else: 
            bf = bf2 = bf3 = np.array([0])
        # Y series
        yf = f[np.char.startswith(f, 'y')]
        if yf.size > 0:
            yf = np.array([i.replace('y' , '') for i in yf])
            yf3 = yf[np.char.endswith(yf, '+++')]
            yf = yf[np.invert(np.char.endswith(yf, '+++'))]
            yf2 = yf[np.char.endswith(yf, '++')]
            yf = yf[np.invert(np.char.endswith(yf, '++'))]
            yf3 = np.array([int(i.replace('+' , '')) for i in yf3])
            yf2 = np.array([int(i.replace('+' , '')) for i in yf2])
            yf = np.array([int(i) for i in yf])
        else: 
            yf = yf2 = yf3 = np.array([0])
        # Claculate continuity per charge
        beta = (((sum((bf[1:]-bf[:-1])==1) + sum((yf[1:]-yf[:-1])==1)) * 0.075) +
                ((sum((bf2[1:]-bf2[:-1])==1) + sum((yf2[1:]-yf2[:-1])==1)) * 0.075) +
                ((sum((bf3[1:]-bf3[:-1])==1) + sum((yf3[1:]-yf3[:-1])==1)) * 0.075))
        # Immonium_ions
        rho = 0
        immonium_ions = [('H', 110.0718), ('Y', 136.0762), ('W', 159.0922), ('M', 104.0534), ('F', 120.0813)]
        immonium_ions = [(i[0],i[1]) for i in immonium_ions if i[1] >= sub_spec[0].min()-(i[1]/((1/ftol)*1E6))]
        for i in immonium_ions:
            if min(abs(sub_spec[0] - i[1])) <= (i[1]/((1/ftol)*1E6)): # immonium ion found
                minloc = np.where(abs(sub_spec[0]-i[1]) == min(abs(sub_spec[0] - i[1])))
                if sub_spec[1][minloc] > 0: # TODO minimum intensity to be considered?
                    if i[0] in seq: # increase rho if immonium in sequence
                        rho += 0.15
                    else: # decrease rho if immonium not in sequence
                        rho -= 0.15
        nm = len(mfrags)
        im = matched_ions
        nt = len(seq) * 2 * 3 # 2 series, 3 charge states # TODO param
        sp = (im * nm * (1 + beta) * (1 + rho)) / nt
    else:
        sp = 0
    return(sp)

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
    omod = []
    opos = []
    for pos, value in modlist:
        peptide = peptide[:pos] + '[' + str(value) + ']' + peptide[pos:]
        omod.append(value)
        opos.append(pos-1)
    return(peptide, omod, opos)

def getTheoMH(sequence, nt, ct, mass,
              m_proton, m_hydrogen, m_oxygen):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    AAs = dict(mass._sections['Aminoacids'])
    MODs = dict(mass._sections['Fixed Modifications'])
    # total_aas = 2*m_hydrogen + m_oxygen
    total_aas = m_proton
    # total_aas += charge*m_proton
    #total_aas += float(MODs['nt']) + float(MODs['ct'])
    if nt:
        total_aas += float(MODs['nt'])
    if ct:
        total_aas += float(MODs['ct'])
    for i, aa in enumerate(sequence):
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
        if aa.lower() in MODs:
            total_aas += float(MODs[aa.lower()])
        # if aa.islower():
        #     total_aas += float(MODs['isolab'])
        # if i in pos:
        #     total_aas += float(mods[pos.index(i)]) TODO: add mod mass outside
    # MH = total_aas - m_proton
    return(total_aas)

def expSpectrum(ions):
    '''
    Prepare experimental spectrum.
    '''
    #ions[0] is mz
    #ions[1] is int
    ions_ZERO = list([0]*len(ions[0]))
    ions_CCU = ions[0] - 0.01
    
    bind = np.array([list(itertools.chain.from_iterable(zip(ions_CCU,ions[0]))),
                     list(itertools.chain.from_iterable(zip(ions_ZERO,ions[1]))),
                     list([0]*len(ions[0])*2),
                     list(np.array(list(itertools.chain.from_iterable(zip(ions_CCU,ions[0])))) + 0.01)
                     ])
    
    spec = np.array([list(itertools.chain.from_iterable(zip(list(bind[0]),list(bind[3])))),
                     list(itertools.chain.from_iterable(zip(list(bind[1]),list(bind[2]))))])
    
    median_rel_int = statistics.median(ions[1])
    std_rel_int = np.std(ions[1], ddof = 1)
    ions_NORM_REL_INT = (ions[1] - median_rel_int) / std_rel_int
    ions_P_REL_INT = scipy.stats.norm.cdf(ions_NORM_REL_INT) #, 0, 1)
    ions = np.array([ions[0], ions[1], ions_ZERO, ions_CCU, ions_NORM_REL_INT, ions_P_REL_INT])
    normspec = ions[1][ions[5]>0.81]
    if len(ions) > 0 and len(normspec) > 0:
        spec_correction = max(ions[1])/statistics.mean(normspec)
    else: spec_correction = 0
    order = np.argsort(np.array(ions)[0])
    ions = [i[order] for i in ions]
    return(spec, ions, spec_correction)

def theoSpectrum(seq, blist, ylist, mods, pos, mass,
                 m_proton, m_hydrogen, m_oxygen, dm=0):
    # blist = [i - 2 for i in blist]
    # ylist = [len(seq)-i for i in ylist]
    ## Y SERIES ##
    outy = []
    # for i in range(1,len(seq)):
    for i in ylist:
        # yn = list(seq[i:])
        yn = list(seq[-i:])
        if i > 0: nt = False
        else: nt = True
        fragy = getTheoMH(yn,nt,True,mass,
                          m_proton,m_hydrogen,m_oxygen) + 2*m_hydrogen + m_oxygen + dm
        outy += [fragy]
    ## B SERIES ##
    outb = []
    # for i in range(1,len(seq)):
    for i in blist:
        bn = list(seq[:i][::-1])
        # bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(bn,True,ct,mass,
                          m_proton,m_hydrogen,m_oxygen) + dm # TODO only add +dm to fragments up until n_pos
        outb += [fragb]
    ## FRAGMENT MATRIX ##
    spec = [outb, outy[::-1]]
    ## ADD FIXED MODS ## # TODO two modes, use mods from config file or input table
    # for i, m in enumerate(mods):
        # bpos = range(0, pos[mods.index(i)]+1)
        # ypos = range(len(seq)-pos[mods.index(i)]-1, len(seq))
        # bpos = pos[i]
        # ypos = len(seq)-pos[i]-1
        # spec[0] = spec[0][:bpos] + [b + m for b in spec[0][bpos:]]
        # spec[1] = spec[1][:ypos] + [y + m for y in spec[1][ypos:]]
    return(spec)

def addMod(spec, dm, pos, len_seq, blist, ylist):
    ## ADD MOD TO SITES ##
    bpos = [i >= pos+1 for i in blist]
    ypos = [i >= len_seq-pos for i in ylist][::-1]
    spec[0] = [spec[0][i]+dm if bpos[i]==True else spec[0][i] for i in list(range(0,len(spec[0])))]
    spec[1] = [spec[1][i]+dm if ypos[i]==True else spec[1][i] for i in list(range(0,len(spec[1])))]
    return(spec)
    
def errorMatrix(mz, theo_spec, m_proton):
    '''
    Prepare ppm-error and experimental mass matrices.
    '''

    theo_spec = theo_spec[0] + theo_spec[1][::-1]
    # theo_spec = np.array([theo_spec]*len(mz))
    theo_spec = np.tile(np.array(np.array(theo_spec)), (len(mz), 1))
    exp = np.transpose(np.array([mz]*len(theo_spec[0])))
    
    ## EXPERIMENTAL MASSES FOR CHARGE 2 ##
    mzs2 = np.transpose([np.array(mz)*2-m_proton]*(len(exp[0])))
    ## EXPERIMENTAL MASSES FOR CHARGE 3 ##
    mzs3 = np.transpose([np.array(mz)*3 - m_proton*2]*(len(exp[0])))
    ## PPM ERRORS ##
    terrors = np.absolute(np.divide(np.subtract(exp, theo_spec), theo_spec)*1000000)
    terrors2 = np.absolute(np.divide(np.subtract(mzs2, theo_spec), theo_spec)*1000000)
    terrors3 = np.absolute(np.divide(np.subtract(mzs3, theo_spec), theo_spec)*1000000)
    
    exp = [i[0] for i in exp]
    
    return(terrors, terrors2, terrors3, exp)

def makeFrags(seq): # TODO: SLOW
    '''
    Name all fragments.
    '''
    bp = bh = yp = yh = []
    # if 'P' in seq[:-1]: # Cannot cut after
    #     # Disallowed b ions
    #     bp = [pos+1 for pos, char in enumerate(seq) if char == 'P']
    #     # Disallowed y ions
    #     yp = [pos for pos, char in enumerate(seq[::-1]) if char == 'P']
    # if 'H' in seq[1:]: # Cannot cut before
    #     # Disallowed b ions
    #     bh = [pos for pos, char in enumerate(seq) if char == 'H']
    #     # Disallowed y ions
    #     yh = [pos+1 for pos, char in enumerate(seq[::-1]) if char == 'H']
    seq_len = len(seq)
    blist = list(range(1,seq_len+1))
    blist = [i for i in blist if i not in bp + bh]
    ylist = list(range(1,seq_len+1))[::-1]
    ylist = [i for i in ylist if i not in yp + yh]
    frags = np.array([["b" + str(i) for i in blist] + ["y" + str(i) for i in ylist],
                      ["b" + str(i) + "++" for i in blist] + ["y" + str(i) + "++" for i in ylist],
                      ["b" + str(i) + "+++" for i in blist] + ["y" + str(i) + "+++" for i in ylist],
                      ["b" + str(i) + "*" for i in blist] + ["y" + str(i) + "*" for i in ylist],
                      ["b" + str(i) + "*++" for i in blist] + ["y" + str(i) + "*++" for i in ylist],
                      ["b" + str(i) + "*+++" for i in blist] + ["y" + str(i) + "*+++" for i in ylist]])
    return(frags, blist, ylist)

def assignIons(theo_spec, dm_theo_spec, frags, dm, mass):
    theo_spec = np.array(theo_spec[0] + theo_spec[1][::-1])
    m_proton = mass.getfloat('Masses', 'm_proton')
    # if dm == 0:
    #     frags = frags[:3]
    #     assign = np.array([#frags[0],
    #                        theo_spec, (theo_spec+m_proton)/2, (theo_spec+2*m_proton)/3])
    #     c_assign_ions = itertools.cycle([i for i in list(range(1,len(assign[0])+1))] + [i for i in list(range(1,len(assign[0])+1))[::-1]])
    #     c_assign = np.array([assign[0:].flatten(),
    #                          #frags[:5].flatten(),
    #                          [next(c_assign_ions) for i in range(len(assign[0:].flatten()))]])
    #                          #[1]*len(assign[0]) + [2]*len(assign[0]) + [3]*len(assign[0]) + [1]*len(assign[0]) + [2]*len(assign[0])])
    # else:
    frags = frags[3:]
    dm_theo_spec = np.array(dm_theo_spec[0] + dm_theo_spec[1][::-1])
    assign = np.array([#frags[0],
                       dm_theo_spec, (dm_theo_spec+m_proton)/2, (dm_theo_spec+m_proton)/3])
    c_assign_ions = itertools.cycle([i for i in list(range(1,len(assign[0])+1))] + [i for i in list(range(1,len(assign[0])+1))[::-1]])
    c_assign = np.array([assign[0:].flatten(),
                         #frags[:5].flatten(),
                         [next(c_assign_ions) for i in range(len(assign[0:].flatten()))]])
                         #[1]*len(assign[0]) + [2]*len(assign[0]) + [3]*len(assign[0]) + [1]*len(assign[0]) + [2]*len(assign[0])])
    return(c_assign, frags.flatten())

def fragCheck(plainseq, blist, ylist, dm_pos):
    ballowed = (['b'+str(i)+'*' if i >= dm_pos+1 else 'b'+str(i) for i in blist] +
                ['b'+str(i)+'*++' if i >= dm_pos+1 else 'b'+str(i)+'++' for i in blist] +
                ['b'+str(i)+'*+++' if i >= dm_pos+1 else 'b'+str(i)+'+++' for i in blist])
    yallowed = (['y'+str(i)+'*' if i >= len(plainseq)-dm_pos else 'y'+str(i) for i in ylist] +
                ['y'+str(i)+'*++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'++' for i in ylist] +
                ['y'+str(i)+'*+++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+++' for i in ylist])
    allowed = ballowed + yallowed
    return(allowed)

def makeAblines(texp, minv, assign, afrags, ions, allowed, tie=51):
    masses = np.array([texp, minv])
    matches = np.array([masses[0][(masses[1]<tie) & ((masses[0]+masses[1])>=0.001)],
                        masses[1][(masses[1]<tie) & ((masses[0]+masses[1])>=0.001)]])
    if len(matches[0]) <= 0:
        proof = np.array([[0],[0],[0]])
        pfrags = np.array([])
        return(proof, pfrags)
    temp_mi = np.repeat(list(matches[0]), len(assign[0]))
    temp_ci1 = np.tile(afrags, len(matches[0]))
    temp_mi1 = np.repeat(list(matches[1]), len(assign[0]))
    temp_ci = np.tile(np.array(assign[0]), len(matches[0]))
    check = abs(temp_mi-temp_ci)/temp_ci*1000000
    if len(check) <= 0:
        proof = np.array([[0],[0],[0]])
        pfrags = np.array([])
        return(proof, pfrags)
    temp_mi = temp_mi[check<=tie]
    proof = np.array([temp_mi,
                      temp_mi1[check<=tie],
                      np.repeat(ions[1][np.isin(ions[0], temp_mi)], np.unique(temp_mi, return_counts=True)[1])])
    pfrags = temp_ci1[check<=tie]
    if len(proof[0]) == 0:
        mzcycle = itertools.cycle([ions[0][0], ions[0][1]])
        proof = np.array([temp_mi,
                          temp_mi1[check<=tie],
                          [next(mzcycle) for i in range(len(temp_mi))]])
        pfrags = temp_ci1[check<=tie]
    # CLEAN UP PFRAGS BY 1) DUPLICATES 2) NONEXISTENT MODIFIED FRAGMENTS
    if len(set(pfrags)) < len(pfrags):
        sorting = pfrags.argsort()
        pfrags = pfrags[sorting]
        proof = np.array([proof[0][sorting], proof[1][sorting], proof[2][sorting]])
        
        proof_mz_groups = np.split(proof[0,:], np.unique(pfrags, return_index=True)[1][1:])
        proof_ppm_groups = np.split(proof[1,:], np.unique(pfrags, return_index=True)[1][1:])
        proof_int_groups = np.split(proof[2,:], np.unique(pfrags, return_index=True)[1][1:])
        for i in range(len(proof_ppm_groups)):
            if len(proof_ppm_groups[i]) > 1:
                amin = np.argmin(proof_ppm_groups[i])
                proof_mz_groups[i] = np.array([proof_mz_groups[i][amin]])
                proof_ppm_groups[i] = np.array([proof_ppm_groups[i][amin]])
                proof_int_groups[i] = np.array([proof_int_groups[i][amin]])
        # acheck = (len(np.concatenate(proof_mz_groups)) == len(np.concatenate(proof_ppm_groups)) == len(np.concatenate(proof_int_groups)))
        proof = np.array([np.concatenate(proof_mz_groups),
                          np.concatenate(proof_ppm_groups),
                          np.concatenate(proof_int_groups)])
    proof[0] = proof[0][proof[0].argsort()]
    proof[1] = proof[1][proof[0].argsort()]
    proof[2] = proof[2][proof[0].argsort()]
    pfrags = np.array(list(dict.fromkeys(pfrags)))
    pfrags = pfrags[proof[0].argsort()]
    fragfilter = [True if i not in allowed else False for i in pfrags]
    pfrags = pfrags[fragfilter]
    proof = np.array([proof[0][fragfilter],
                      proof[1][fragfilter],
                      proof[2][fragfilter]])
    return(proof, pfrags)

def findClosest(dm, dmdf, dmtol, pos):
    cand = [i for i in range(len(dmdf[1])) if dmdf[1][i] > dm-dmtol and dmdf[1][i] < dm+dmtol]
    closest = pd.DataFrame([dmdf[0][cand], dmdf[1][cand], dmdf[2][cand], dmdf[3][cand]]).T
    closest.columns = ['name', 'mass', 'site', 'site_tiebreaker']
    closest = pd.concat([closest, pd.Series({'name':'EXPERIMENTAL', 'mass':dm, 'site':[pos], 'site_tiebreaker':[pos]}).to_frame().T], ignore_index=True)
    return(closest)

def findPos(dm_set, plainseq): # TODO fix sites now that this is array instead of DF
    def _where(sites, plainseq):
        sites = sites.site
        subpos = []
        for s in sites:
            if s == 'Anywhere' or s == 'exp':
                subpos = list(range(0, len(plainseq)))
                break
            elif s == 'Non-modified':
                subpos = [-1]
                break
            elif s == 'N-term':
                subpos = [0]
            elif s == 'C-term':
                subpos = [len(plainseq) - 1]
            else:
                subpos = subpos + list(np.where(np.array(list(plainseq)) == str(s))[0])
        subpos = list(dict.fromkeys(subpos))
        subpos.sort()
        return(subpos)
    dm_set['idx'] = dm_set.apply(_where, plainseq=plainseq, axis=1)
    dm_set = dm_set[dm_set.idx.apply(lambda x: len(x)) > 0]
    return(dm_set)

def miniVseq(sub, plainseq, mods, pos, mass, ftol, dmtol, dmdf, exp_spec, ions,
             spec_correction, m_proton, m_hydrogen, m_oxygen, ttol, tmin):
    ## ASSIGNDB ##
    # assigndblist = []
    # assigndb = []
    ## FRAGMENT NAMES ##
    frags, blist, ylist = makeFrags(plainseq)
    ## DM ##
    exp_pos = 'exp'
    dm_set = findClosest(sub.DM, dmdf, dmtol, exp_pos) # Contains experimental DM
    dm_set = findPos(dm_set, plainseq)
    # if 0 in dm_set.mass.values:
    #     dm_set.at[list(dm_set[dm_set.mass==0].index)[0],'idx'] = [0]
    # else:
    dm_set = pd.concat([dm_set,
                        pd.Series({'name':'Non-modified', 'mass':0, 'site':['Anywhere'], 'site_tiebreaker':['Non-modified'], 'idx':[0]}).to_frame().T], ignore_index=True)
    theo_spec = theoSpectrum(plainseq, blist, ylist, mods, pos, mass,
                             m_proton, m_hydrogen, m_oxygen)
    terrors, terrors2, terrors3, texp = errorMatrix(ions[0], theo_spec, m_proton)
    closest_proof = []
    closest_pfrags = []
    closest_dm = []
    closest_name = []
    closest_pos = []
    closest_tie = []
    for index, row in dm_set.iterrows():
        dm = row.mass
        temp_proof = []
        temp_pfrags = []
        temp_dm = []
        temp_name = []
        temp_pos = []
        tiebreaker = []
        for dm_pos in row.idx:
            allowed = fragCheck(plainseq, blist, ylist, dm_pos)
            ## DM OPERATIONS ##
            if dm_pos == -1: # Non-modified
                dm_theo_spec = theo_spec.copy()
                # dm_theo_spec = [x+dm for x in dm_theo_spec[0]] + [x+dm for x in dm_theo_spec[1]]
            else:
                dm_theo_spec = theo_spec.copy()
                dm_theo_spec = addMod(dm_theo_spec, dm, dm_pos, len(plainseq), blist, ylist)
            ## ASSIGN IONS WITHIN SPECTRA ##
            assign, afrags = assignIons(theo_spec, dm_theo_spec, frags, dm, mass)
            # TODO check that we don't actually need to calculate the proof (adds PPM) (check this by making sure minv is also equal and assign and minv are the only things that can change the proof)
            ## PPM ERRORS ##
            if dm != 0:
                dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(ions[0], dm_theo_spec, m_proton)
                if sub.Charge == 2:
                    ppmfinal = pd.DataFrame(np.array([terrors, terrors2,
                                                      dmterrors, dmterrors2]).min(0))
                elif sub.Charge < 2:
                    ppmfinal = pd.DataFrame(np.array([terrors,
                                                      dmterrors]).min(0))
                elif sub.Charge >= 3:
                    ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3,
                                                      dmterrors, dmterrors2, dmterrors3]).min(0))
                else:
                    sys.exit('ERROR: Invalid charge value!')
            else:
                if sub.Charge == 2:
                    ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
                elif sub.Charge < 2:
                    ppmfinal = pd.DataFrame(np.array([terrors]).min(0))
                elif sub.Charge >= 3:
                    ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3]).min(0))
                else:
                    sys.exit('ERROR: Invalid charge value!')
            minv = list(ppmfinal.min(axis=1))
            ## ABLINES ##
            proof, pfrags = makeAblines(texp, minv, assign, afrags, ions, allowed)
            if pfrags.size > 0:
                proof[2] = proof[2] * spec_correction
                proof[2][proof[2] > exp_spec[1].max()] = exp_spec[1].max() - 3
                pfrags = pfrags[proof[1] <= ftol]
                proof = np.array([proof[0][proof[1] <= ftol],
                                  proof[1][proof[1] <= ftol],
                                  proof[2][proof[1] <= ftol]])
                # Choose only lowest error match for each peak
                if len(np.unique(proof[0])) != len(proof[0]):
                    #mask = np.unique(proof[0], return_inverse=True)[1]
                    split0 = np.split(proof[0], np.unique(proof[0], return_index=True)[1][1:])
                    split1 = np.split(proof[1], np.unique(proof[0], return_index=True)[1][1:])
                    split2 = np.split(proof[2], np.unique(proof[0], return_index=True)[1][1:])
                    splitf = np.split(pfrags, np.unique(proof[0], return_index=True)[1][1:])
                    mins = [np.argmin(i) for i in split1]
                    minl = range(0,len(mins))
                    proof = np.array([[split0[i][mins[i]] for i in minl],
                                      [split1[i][mins[i]] for i in minl],
                                      [split2[i][mins[i]] for i in minl]])
                    pfrags = np.array([splitf[i][mins[i]] for i in minl])
            tiebreaker.append(''.join(list(pfrags)))
            temp_proof.append(proof)
            temp_pfrags.append(pfrags)
            temp_dm.append(dm)
            temp_name.append(row['name'])
            temp_pos.append(dm_pos)
        ## TIE-BREAKER ##
        if len(set(tiebreaker)) <= len(plainseq)-len(plainseq)*tmin:
            temp_tie = [True if tiebreaker.count(tiebreaker[i]) > 1 else False for i in range(len(tiebreaker))]
            for i in range(len(row.idx)):
                dm_pos = row.idx[i]
                ## DM OPERATIONS ##
                if dm_pos == -1: # Non-modified
                    dm_theo_spec = theo_spec.copy()
                    # dm_theo_spec = [x+dm for x in dm_theo_spec[0]] + [x+dm for x in dm_theo_spec[1]]
                else:
                    dm_theo_spec = theo_spec.copy()
                    dm_theo_spec = addMod(dm_theo_spec, dm, dm_pos, len(plainseq), blist, ylist)
                ## ASSIGN IONS WITHIN SPECTRA ##
                assign, afrags = assignIons(theo_spec, dm_theo_spec, frags, dm, mass)
                # TODO check that we don't actually need to calculate the proof (adds PPM) (check this by making sure minv is also equal ans assign and minv are the only things that can change the proof)
                ## PPM ERRORS ##
                # if dm != 0:
                dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(ions[0], dm_theo_spec, m_proton)
                if sub.Charge == 2:
                    ppmfinal = pd.DataFrame(np.array([dmterrors, dmterrors2]).min(0))
                elif sub.Charge < 2:
                    ppmfinal = pd.DataFrame(np.array([dmterrors]).min(0))
                elif sub.Charge >= 3:
                    ppmfinal = pd.DataFrame(np.array([dmterrors, dmterrors2, dmterrors3]).min(0))
                else:
                    sys.exit('ERROR: Invalid charge value!')
                # else:
                #     if sub.Charge == 2:
                #         ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
                #     elif sub.Charge < 2:
                #         ppmfinal = pd.DataFrame(np.array([terrors]).min(0))
                #     elif sub.Charge >= 3:
                #         ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3]).min(0))
                #     else:
                #         sys.exit('ERROR: Invalid charge value!')
                minv = list(ppmfinal.min(axis=1))
                ## ABLINES ##
                proof, pfrags = makeAblines(texp, minv, assign, afrags, ions, allowed, ftol+ttol)
                if pfrags.size > 0:
                    proof[2] = proof[2] * spec_correction
                    proof[2][proof[2] > exp_spec[1].max()] = exp_spec[1].max() - 3
                    pfrags = pfrags[proof[1] <= ftol+ttol]
                    proof = np.array([proof[0][proof[1] <= ftol+ttol],
                                      proof[1][proof[1] <= ftol+ttol],
                                      proof[2][proof[1] <= ftol+ttol]])
                    # Choose only lowest error match for each peak
                    if len(np.unique(proof[0])) != len(proof[0]):
                        #mask = np.unique(proof[0], return_inverse=True)[1]
                        split0 = np.split(proof[0], np.unique(proof[0], return_index=True)[1][1:])
                        split1 = np.split(proof[1], np.unique(proof[0], return_index=True)[1][1:])
                        split2 = np.split(proof[2], np.unique(proof[0], return_index=True)[1][1:])
                        splitf = np.split(pfrags, np.unique(proof[0], return_index=True)[1][1:])
                        mins = [np.argmin(i) for i in split1]
                        minl = range(0,len(mins))
                        proof = np.array([[split0[i][mins[i]] for i in minl],
                                          [split1[i][mins[i]] for i in minl],
                                          [split2[i][mins[i]] for i in minl]])
                        pfrags = np.array([splitf[i][mins[i]] for i in minl])
                temp_proof[i] = proof
                temp_pfrags[i] = pfrags
                temp_dm[i] = dm
                temp_name[i] = row['name']
                temp_pos[i] = dm_pos
        else:
            temp_tie = [False]*len(temp_proof)
        closest_proof += temp_proof
        closest_pfrags += temp_pfrags
        closest_dm += temp_dm
        closest_name += temp_name
        closest_pos += temp_pos
        closest_tie += temp_tie
    return(closest_proof, closest_pfrags, closest_dm, closest_name, closest_pos, closest_tie)

def parallelFragging(query, parlist):
    m_proton = parlist[4]
    scan = query.scannum
    charge = query.charge
    MH = query.precursor_neutral_mass + (m_proton)
    plain_peptide = query.peptide
    dmdf = parlist[3]
    if pd.isnull(query.modification_info):
        sequence = plain_peptide
        mod = []
        pos = []
    else:
        sequence, mod, pos = insertMods(plain_peptide, query.modification_info)
    if parlist[9]=="mzml":
        spectrum = parlist[11][np.where(np.array(parlist[10])==scan)[0][0]]
    else:
        spectrum = query.spectrum
    dm = query.massdiff
    # TODO use calc neutral mass?
    # Make a Vseq-style query
    sub = pd.Series([scan, charge, MH, sequence, spectrum, dm],
                    index = ["FirstScan", "Charge", "MH", "Sequence", "Spectrum", "DM"])
    exp_spec, exp_ions, spec_correction = expSpectrum(sub.Spectrum)
    proof, pfrags, dm, name, position, tie = miniVseq(sub, plain_peptide, mod, pos,
                                                 parlist[0], parlist[1], parlist[2],
                                                 parlist[3], exp_spec, exp_ions, spec_correction,
                                                 parlist[4], parlist[5], parlist[6],
                                                 parlist[7], parlist[8])
    #matched_ions_names = pfrags.copy()
    # TODO: always get a Non-modified score
    # Remove cases where a DM is tried but no modified fragments have been matched
    check = []
    for i in range(0, len(pfrags)):
        if name[i] not in ['EXPERIMENTAL', 'Non-modified']:
            if ''.join(pfrags[i]).find('*') == -1:
                check += [False]
            else:
                check += [True]
        else:
            check += [True]
    proof, pfrags, dm, name, position, tie = list(itertools.compress(proof, check)), list(itertools.compress(pfrags, check)), list(itertools.compress(dm, check)), list(itertools.compress(name, check)), list(itertools.compress(position, check)), list(itertools.compress(tie, check))
    
    hyperscores = []
    hyperscores_label = []
    check = []
    hss = []
    ufrags = []
    for i in list(range(0, len(dm))):
        total = proof[i][0].sum() + proof[i][2].sum()
        if total in check:
            hscore = hss[check.index(total)]
            frags = ufrags[check.index(total)]
            pfrags[i] = np.unique(np.array([f.replace('*' , '') for f in pfrags[i]]))
        else:
            if pfrags[i].size > 0:
                if tie[i]:
                    hscore, isum = hyperscore(sub.Charge, exp_ions, proof[i], pfrags[i], parlist[1]+parlist[8])
                else:
                    hscore, isum = hyperscore(sub.Charge, exp_ions, proof[i], pfrags[i], parlist[1])
            else:
                hscore = isum = 0
            #pfrags[i] = np.array([f.replace('+' , '').replace('*' , '') for f in pfrags[i]])
            pfrags[i] = np.unique(np.array([f.replace('*' , '') for f in pfrags[i]]))
            frags = len(np.unique(np.array([f.replace('+' , '') for f in pfrags[i]])))
            check += [total]
            hss += [hscore]
            ufrags += [frags]
        hyperscores += [[dm[i], position[i], frags, hscore, i, isum]]
        hyperscores_label += [name[i]]
    # best = hyperscores[[i[4] for i in hyperscores].index(max([i[4] for i in hyperscores]))]
    hyperscores = np.transpose(np.array(hyperscores))
    hyperscores_label = np.array(hyperscores_label)
    nm = np.array([hyperscores[0][hyperscores_label == 'Non-modified'],
                   hyperscores[1][hyperscores_label == 'Non-modified'],
                   hyperscores[2][hyperscores_label == 'Non-modified'],
                   hyperscores[3][hyperscores_label == 'Non-modified']])
    nm = np.array([nm[0][nm[3]==nm[3].max()][0],
                     nm[1][nm[3]==nm[3].max()][0],
                     nm[2][nm[3]==nm[3].max()][0],
                     nm[3][nm[3]==nm[3].max()][0]])
    # if not (query.massdiff-parlist[2] <= 0 <= query.massdiff+parlist[2]):
    #     hyperscores = np.array([np.delete(hyperscores[0], hyperscores_label == 'Non-modified'),
    #                      np.delete(hyperscores[1], hyperscores_label == 'Non-modified'),
    #                      np.delete(hyperscores[2], hyperscores_label == 'Non-modified'),
    #                      np.delete(hyperscores[3], hyperscores_label == 'Non-modified'),
    #                      np.delete(hyperscores[4], hyperscores_label == 'Non-modified'),
    #                      np.delete(hyperscores[5], hyperscores_label == 'Non-modified')])
    #     hyperscores_label = np.delete(hyperscores_label, hyperscores_label == 'Non-modified')
    best = np.array([hyperscores[0][hyperscores[3]==hyperscores[3].max()],
                     hyperscores[1][hyperscores[3]==hyperscores[3].max()],
                     hyperscores[2][hyperscores[3]==hyperscores[3].max()],
                     hyperscores[3][hyperscores[3]==hyperscores[3].max()],
                     hyperscores[4][hyperscores[3]==hyperscores[3].max()],
                     hyperscores[5][hyperscores[3]==hyperscores[3].max()]])
    best_label = hyperscores_label[hyperscores[3]==hyperscores[3].max()]
    if len(best[0]) > 1:
        # In case of tie, keep most matched_ions
        best_label = best_label[best[2]==best[2].max()]
        best = np.array([best[0][best[2]==best[2].max()],
                         best[1][best[2]==best[2].max()],
                         best[2][best[2]==best[2].max()],
                         best[3][best[2]==best[2].max()],
                         best[4][best[2]==best[2].max()],
                         best[5][best[2]==best[2].max()]])
        # In case of tie, keep highest intensity
        if len(best[0]) > 1:
            best_label = best_label[best[5]==best[5].max()]
            best = np.array([best[0][best[5]==best[5].max()],
                             best[1][best[5]==best[5].max()],
                             best[2][best[5]==best[5].max()],
                             best[3][best[5]==best[5].max()],
                             best[4][best[5]==best[5].max()],
                             best[5][best[5]==best[5].max()]])
        if len(best[0]) > 1:
            # Prefer theoretical rather than experimental # TODO
            if 0 < (best_label == 'EXPERIMENTAL').sum() < len(best_label):
                best = np.array([np.delete(best[0], best_label == 'EXPERIMENTAL'),
                                 np.delete(best[1], best_label == 'EXPERIMENTAL'),
                                 np.delete(best[2], best_label == 'EXPERIMENTAL'),
                                 np.delete(best[3], best_label == 'EXPERIMENTAL'),
                                 np.delete(best[4], best_label == 'EXPERIMENTAL'),
                                 np.delete(best[5], best_label == 'EXPERIMENTAL')])
                best_label = np.delete(best_label, best_label == 'EXPERIMENTAL')
            # Prefer NM rather than modified
            if 0 < (best_label == 'Non-modified').sum() < len(best_label):
                best = np.array([np.delete(best[0], best_label != 'Non-modified'),
                                 np.delete(best[1], best_label != 'Non-modified'),
                                 np.delete(best[2], best_label != 'Non-modified'),
                                 np.delete(best[3], best_label != 'Non-modified'),
                                 np.delete(best[4], best_label != 'Non-modified'),
                                 np.delete(best[5], best_label != 'Non-modified')])
                best_label = np.delete(best_label, best_label != 'Non-modified')
            if len(best[0]) > 1:
                # Prefer cases where the AA location is possible according to UNIMOD
                pos_check = np.array([plain_peptide[int(j)]+str(int(bool(j))) if j==0 or j==len(plain_peptide)-1 else plain_peptide[int(j)] for j in best[1]])
                name_check = np.array([dmdf[3][dmdf[1]==k][0] if (k!=0)&(len(dmdf[3][dmdf[1]==k])>0) else string.ascii_uppercase for k in best[0]])
                name_check[np.where(name_check=='C-term')] = plain_peptide[-1] # TODO consider AA in other positions that match the first/last AA
                name_check[np.where(name_check=='N-term')] = plain_peptide[0]
                bool_check = [True if len(set(pos_check[l]).intersection(name_check[l]))>0 else False for l in range(len(pos_check))]
                if sum(bool_check) > 0:
                    best = np.array([best[m][bool_check] for m in range(len(best))])
                    best_label = best_label[bool_check]
        # Keep closest to experimental # TODO mark several possible locations (same score)
        # best = np.array([best[0][0], best[1][0], best[2][0], best[3][0], best[4][0], best[5][0]])
        best_index = int(np.where(abs(best[0]-sub.DM) == min(abs(best[0]-sub.DM)))[0][0])
        best = np.array([best[0][best_index], best[1][best_index], best[2][best_index],
                         best[3][best_index], best[4][best_index], best[5][best_index]])
        best_label = np.array(best_label[best_index])
    elif len(best[0]) == 1:
        best = np.array([best[0][0], best[1][0], best[2][0], best[3][0], best[4][0], best[5][0]])
        best_label = np.array(best_label[0])
    exp = np.array([hyperscores[0][hyperscores_label == 'EXPERIMENTAL'],
                   hyperscores[1][hyperscores_label == 'EXPERIMENTAL'],
                   hyperscores[2][hyperscores_label == 'EXPERIMENTAL'],
                   hyperscores[3][hyperscores_label == 'EXPERIMENTAL']])
    exp = np.array([exp[0][exp[3]==exp[3].max()][0],
                     exp[1][exp[3]==exp[3].max()][0],
                     exp[2][exp[3]==exp[3].max()][0],
                     exp[3][exp[3]==exp[3].max()][0]])
    try:
        best_label = str(best_label[0])
    except IndexError:
        best_label = str(best_label)
    sp = spscore(sub.Spectrum, best[5], parlist[1], query.peptide, pfrags[int(best[4])])
    #matched_ions_names = matched_ions_names[np.where((hyperscores[0]==best[0])&(hyperscores[1]==best[1])&(hyperscores[2]==best[2])&(hyperscores[3]==best[3])&(hyperscores[4]==best[4])&(hyperscores[5]==best[5]))[0][0]]
    best_pos = plain_peptide[int(best[1])]+str(int(best[1]+1))
    if best_label == 'Non-modified':
        best_pos = ''
    return([MH, float(best[0]), sequence, int(best[2]), float(best[3]), best_label,
            float(exp[0]), float(exp[3]), best_pos,
            sp, int(exp[2]), float(nm[3]), int(nm[2]), float(best[5])])

def makeSummary(df, outpath, infile, raw, dmlist, startt, endt, decoy):
    
    smods = df.REFRAG_name.value_counts()
    smods = smods[smods.index!='EXPERIMENTAL']
    lsmods = []
    for i in range(0, len(smods)):
        lsmods += [str(list(smods)[i]), '\t', str(list(smods.index)[i]), '\n']
    lsmods = ''.join(lsmods)
        
    summary = (
    "DATE\t{date}\n"
    "FILE\t{infile}\n"
    "RAW\t{raw}\n"
    "DMLIST\t{dmlist}\n"
    "SEARCH TIME\t{time}\n"
    "TOTAL PSMs\t{total}\n"
    "TARGET PSMs\t{target}\n"
    "REFRAGGED PSMs\t{refrag}\t({perc}% of total)\n"
    "REFRAGGED TARGET PSMs\t{refragt}\t({perct}% of targets)\n\n"
    "THEORETICAL MODIFICATIONS FREQUENCY\n\n"
    "{smods}").format(date=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    infile=str(infile),
    raw=str(raw),
    dmlist=str(dmlist),
    time=str(endt-startt),
    total=str(len(df)),
    target=str(len(df[~df.protein.str.startswith(decoy)])),
    refrag=str(len(df[df.REFRAG_name!='EXPERIMENTAL'])),
    perc=str(round(len(df[df.REFRAG_name!='EXPERIMENTAL'])/len(df)*100,2)),
    refragt=str(len(df[(df.REFRAG_name!='EXPERIMENTAL')&(~df.protein.str.startswith(decoy))])),
    perct=str(round(len(df[(df.REFRAG_name!='EXPERIMENTAL')&(~df.protein.str.startswith(decoy))])/len(df[~df.protein.str.startswith(decoy)])*100,2)),
    smods=lsmods)
    
    with open(outpath, 'w') as f:
        f.write(summary)
    
    return

def main(args):
    '''
    Main function
    '''
    # Parameters
    chunks = int(mass._sections['Search']['batch_size'])
    ftol = float(mass._sections['Search']['f_tol'])
    ttol = float(mass._sections['Search']['t_tol'])
    tmin = float(mass._sections['Search']['t_min'])
    dmtol = float(mass._sections['Search']['dm_tol'])
    decoy_prefix = str(mass._sections['Search']['decoy_prefix'])
    top_n = int(mass._sections['Spectrum Processing']['top_n'])
    bin_top_n = eval(str(mass._sections['Spectrum Processing']['bin_top_n']).title())
    min_ratio = float(mass._sections['Spectrum Processing']['min_ratio'])
    min_frag_mz = float(mass._sections['Spectrum Processing']['min_fragment_mz'])
    max_frag_mz = float(mass._sections['Spectrum Processing']['max_fragment_mz'])
    deiso = eval(str(mass._sections['Spectrum Processing']['deisotope']).title())
    m_proton = mass.getfloat('Masses', 'm_proton')
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    
    checked = checkParams(mass)
    if checked != 0:
        sys.exit("ERROR: Invalid parameters.")
    
    infiles, rawfiles, rawbase = preProcess(args)
        
    for f in infiles:
        if os.path.basename(f).split(sep=".")[0] in rawbase:
            infile = f
            rawfile = rawfiles[rawbase.index(os.path.basename(f).split(sep=".")[0])]
        else:
            logging.info("ERROR: No MS data file (mzML or MGF) found for MSFragger file " + str(f) + "\nSkipping...")
            continue

        # Read results file from MSFragger
        if len(args.dia) > 0:
            logging.info("Reading MSFragger file (" + str(os.path.basename(Path(infile[0:-8] + infile[-4:]))) + ")...")
            try: df = pd.read_csv(Path(infile[0:-8] + infile[-4:]), sep="\t")
            except pd.errors.ParserError:
                sep = '\t'
                coln = pd.read_csv(Path(infile[0:-8] + infile[-4:]), sep=sep, nrows=1).columns.tolist()
                max_columns = max(open(Path(infile[0:-8] + infile[-4:]), 'r'), key = lambda x: x.count(sep)).count(sep) + 1
                coln += ['extra_column_' + str(i) for i in range(0,(max_columns-len(coln)))]
                df = pd.read_csv(Path(infile[0:-8] + infile[-4:]), header=None, sep=sep, skiprows=1, names=coln)
        else:
            logging.info("Reading MSFragger file (" + str(os.path.basename(Path(infile))) + ")...")
            try: df = pd.read_csv(Path(infile), sep="\t")
            except pd.errors.ParserError:
                sep = '\t'
                coln = pd.read_csv(Path(infile), sep=sep, nrows=1).columns.tolist()
                max_columns = max(open(Path(infile), 'r'), key = lambda x: x.count(sep)).count(sep) + 1
                coln += ['extra_column_' + str(i) for i in range(0,(max_columns-len(coln)))]
                df = pd.read_csv(Path(infile), header=None, sep=sep, skiprows=1, names=coln)
        if len(args.scanrange) > 0:
            logging.info("Filtering scan range " + str(args.scanrange[0]) + "-" + str(args.scanrange[1]) + "...")
            df = df[(df.scannum>=args.scanrange[0])&(df.scannum<=args.scanrange[1])]
        logging.info("\t" + str(len(df)) + " lines read.")
        if len(df) < 1:
            logging.error("No scans to search. Stopping.")
            sys.exit()
        # Read raw file
        msdata, mode, index2 = readRaw(Path(rawfile))
        # Read DM file
        logging.info("Reading DM file (" + str(os.path.basename(Path(args.dmfile))) + ")...")
        dmdf = pd.read_csv(Path(args.dmfile), sep="\t") # TODO check for duplicates (when both DM and SITE is the same)
        dmdf.columns = ["name", "mass", "site", "site_tiebreaker"]
        dmdf.site = dmdf.site.apply(literal_eval)
        dmdf.site = dmdf.apply(lambda x: list(dict.fromkeys(x.site)), axis=1)
        dmdf = dmdf.T.to_numpy()
        dmdf[3] = np.array([''.join(set(''.join(literal_eval(i)).replace('N-term', '0').replace('C-term', '1'))) for i in dmdf[3]]) # Nt = 0, Ct = 1
        logging.info("\t" + str(len(dmdf[0])) + " theoretical DMs read.")
        # Prepare to parallelize
        logging.info("Refragging...")
        logging.info("\t" + "Locating scans...")
        starttime = datetime.now()
        if mode == "mzml":
            spectra = msdata.getSpectra()
            spectra_n = np.array([int(s.getNativeID().split("=")[-1]) for s in spectra])
            if len(args.scanrange) > 0:
                logging.info("Filtering scan range " + str(args.scanrange[0]) + "-" + str(args.scanrange[1]) + "...")
                spectra = spectra[np.where((np.array(spectra_n)>=0)&(np.array(spectra_n)<=args.scanrange[0]))[0][0]:np.where((np.array(spectra_n)>=0)&(np.array(spectra_n)<=args.scanrange[-1]))[0][-1]]
                spectra_n = spectra_n[np.where((np.array(spectra_n)>=0)&(np.array(spectra_n)<=args.scanrange[0]))[0][0]:np.where((np.array(spectra_n)>=0)&(np.array(spectra_n)<=args.scanrange[-1]))[0][-1]]
            peaks = [s.get_peaks() for s in spectra]
            ions = [np.array([p[0], p[1]]) for p in peaks]
            ions0 = [np.array(p[0]) for p in peaks]
            ions1 = [np.array(p[1]) for p in peaks]
            # Remove peaks below min_ratio
            logging.info("\t" + "Filter by ratio...")
            cutoff1 = [i/max(i) >= min_ratio for i in ions1]
            ions0 = [ions0[i][cutoff1[i]] for i in range(len(ions))]
            ions1 = [ions1[i][cutoff1[i]] for i in range(len(ions))]
            # Return only top N peaks
            logging.info("\t" + "Filter top N...")
            cutoff1 = [i >= i[np.argsort(i)[len(i)-top_n]] if len(i)>top_n else i>0 for i in ions1]
            ions0 = [ions0[i][cutoff1[i]] for i in range(len(ions))]
            ions1 = [ions1[i][cutoff1[i]] for i in range(len(ions))]
            ions = [np.array([ions0[i],ions1[i]]) for i in range(len(ions))]
            # # Duplicate m/z measurement
            logging.info("\t" + "Filter duplicate m/z measurements...")
            check = [len(np.unique(i)) != len(i) for i in ions0]
            for i in range(len(check)):
                if check[i] == True:
                    temp = ions[i].copy()
                    temp = pd.DataFrame(temp).T
                    temp = temp[temp.groupby(0)[1].rank(ascending=False)<2]
                    temp.drop_duplicates(subset=0, inplace=True)
                    ions[i] = np.array(temp.T)
            cc = 0
            for i in ions:
                if len(i[0])<2:
                    cc += 1
            logging.info("\t" + str(cc) + " empty spectra...")
        else:
            df.scannum = df.scannum.astype(int)
            df["spectrum"] = df.apply(lambda x: locateScan(x.scannum, mode, msdata,
                                                           0, 0, index2, top_n,
                                                           bin_top_n, min_ratio,
                                                           min_frag_mz, max_frag_mz,
                                                           m_proton, deiso, x.charge),
                                      axis=1)
        indices, rowSeries = zip(*df.iterrows())
        rowSeries = list(rowSeries)
        tqdm.pandas(position=0, leave=True)
        if len(df) <= chunks:
            chunks = math.ceil(len(df)/args.n_workers)
        if mode == "mzml":
            parlist = [mass, ftol, dmtol, dmdf, m_proton, m_hydrogen, m_oxygen, ttol, tmin, mode, spectra_n, ions]
        else:
            parlist = [mass, ftol, dmtol, dmdf, m_proton, m_hydrogen, m_oxygen, ttol, tmin, mode]
        logging.info("\tBatch size: " + str(chunks) + " (" + str(math.ceil(len(df)/chunks)) + " batches)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            refrags = list(tqdm(executor.map(parallelFragging,
                                             rowSeries,
                                             itertools.repeat(parlist),
                                             chunksize=chunks),
                                total=len(rowSeries)))
        if 'spectrum' in df.columns:
            df = df.drop('spectrum', axis = 1)
        df['templist'] = refrags
        df['REFRAG_MH'] = pd.DataFrame(df.templist.tolist()).iloc[:, 0]. tolist()
        df['REFRAG_exp_MZ'] = (df.REFRAG_MH + (df.charge-1)*m_proton) / df.charge
        df['REFRAG_exp_DM'] = pd.DataFrame(df.templist.tolist()).iloc[:, 6]. tolist()
        df['REFRAG_exp_ions_matched'] = pd.DataFrame(df.templist.tolist()).iloc[:, 10]. tolist()
        df['REFRAG_exp_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 7]. tolist()
        df['REFRAG_nm_ions_matched'] = pd.DataFrame(df.templist.tolist()).iloc[:, 12]. tolist()
        df['REFRAG_nm_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 11]. tolist()
        df['REFRAG_DM'] = pd.DataFrame(df.templist.tolist()).iloc[:, 1]. tolist()
        df['REFRAG_site'] = pd.DataFrame(df.templist.tolist()).iloc[:, 8]. tolist()
        df['REFRAG_sequence'] = pd.DataFrame(df.templist.tolist()).iloc[:, 2]. tolist()
        df['REFRAG_ions_matched'] = pd.DataFrame(df.templist.tolist()).iloc[:, 3]. tolist()
        df['REFRAG_sum_intenisty'] = pd.DataFrame(df.templist.tolist()).iloc[:, 13]. tolist()
        df['REFRAG_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 4]. tolist()
        df['REFRAG_name'] = pd.DataFrame(df.templist.tolist()).iloc[:, 5]. tolist()
        df['REFRAG_sp_score'] = pd.DataFrame(df.templist.tolist()).iloc[:, 9]. tolist()
        df = df.drop('templist', axis = 1)
        try:
            refragged = len(df)-df.REFRAG_name.value_counts()['EXPERIMENTAL']
        except KeyError:
            refragged = len(df)
        prefragged = round((refragged/len(df))*100,2)
        logging.info("\t" + str(refragged) + " (" + str(prefragged) + "%) refragged PSMs.")
        endtime = datetime.now()
        logging.info("Writing output file...")
        outpath = Path(os.path.splitext(infile)[0] + "_REFRAG.tsv")
        if len(args.scanrange) > 0:
            outpath = Path(os.path.splitext(infile)[0] + "_" +
                           str(args.scanrange[0]) + "-" + str(args.scanrange[1]) +
                           "_REFRAG.tsv")
        df.to_csv(outpath, index=False, sep='\t', encoding='utf-8')
        logging.info("Done.")
        logging.info("Writing summary file...")
        outsum = Path(os.path.splitext(infile)[0] + "_SUMMARY.tsv")
        if len(args.scanrange) > 0:
            outsum = Path(os.path.splitext(infile)[0] + "_" +
                          str(args.scanrange[0]) + "-" + str(args.scanrange[1]) +
                          "_SUMMARY.tsv")
        makeSummary(df, outsum, infile, rawfile, args.dmfile, starttime, endtime, decoy_prefix)
        logging.info("Done.")
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
    
    # TODO parameter: exclude DM range (consider as NM)= default (-3, 0)
    parser.add_argument('-i',  '--infile', required=True, help='MSFragger results file')
    parser.add_argument('-r',  '--rawfile', required=True, help='MS Data file (MGF or MZML)')
    parser.add_argument('-d',  '--dmfile', required=True, help='DeltaMass file')
    parser.add_argument('-a', '--dia', default=[], help='DIA mode (looks for MS Data files with _chN suffix)',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-s', '--scanrange', default=[], help='Scan range to search',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-w',  '--n_workers', type=int, default=os.cpu_count(), help='Number of threads/n_workers')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    if Path(args.config).is_file():
        mass.read(args.config)
    else:
        sys.exit('ERROR: Cannot find configuration file at' + str(args.config))
    # if something is changed, write a copy of ini
    if mass.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.infile) + '/ReFrag.ini', 'w') as newconfig:
            mass.write(newconfig)

    # logging debug level. By default, info level
    if os.path.isdir(args.infile):
        log_file = os.path.join(args.infile + '/ReFrag.log')
        log_file_debug = os.path.join(args.infile + '/ReFrag_debug.log')
    else:
        log_file = args.infile[:-4] + '_ReFrag.log'
        log_file_debug = args.infile[:-4] + '_ReFrag_debug.log'
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
        
    if len(args.scanrange) > 2:
        sys.exit('ERROR: Scan range list can only contain two elements (start and end)')

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')