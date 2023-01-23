# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:52 2022

@author: alaguillog
"""

from ast import literal_eval
import argparse
import concurrent.futures
import configparser
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
import sys
import shutup
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'
shutup.please()

def readRaw(msdata):
    if os.path.splitext(msdata)[1].lower() == ".mzml":
        mode = "mzml"
        logging.info("Reading mzML file...")
        fr_ns = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(str(msdata), fr_ns)
        index2 = 0
        logging.info("\t" + str(fr_ns.getNrSpectra()) + " spectra read.")
    elif os.path.splitext(msdata)[1].lower() == ".mgf":
        mode = "mgf"
        logging.info("Reading MGF file...")
        fr_ns = pd.read_csv(msdata, header=None)
        index2 = fr_ns.to_numpy() == 'END IONS'
        logging.info("\t" + str(sum(fr_ns[0].str[:4]=="SCAN")) + " spectra read.")
        # logging.info("\t" + str(fr_ns[0].str.count('SCANS').sum()) + " spectra read.") # VERY SLOW
    else:
        logging.info("MS Data file extension not recognized!")
        sys.exit()
    return(fr_ns, mode, index2)

def locateScan(scan, mode, fr_ns, index2):
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
            ions = fr_ns.iloc[index1+1:index3,:]
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
    elif mode == "mzml":
        try:
            s = fr_ns.getSpectrum(scan-1)
        except AssertionError or OverflowError:
            logging.info("\tERROR: Scan number " + str(scan) + " not found in mzML file.")
            sys.exit()
        ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
        ions.columns = ["MZ", "INT"]
    return(ions)

def hyperscore(ions, proof, ftol=50): # TODO play with number of ions # if modified frag present, don't consider non-modified?
    ## 1. Normalize intensity to 10^5
    norm = (ions.INT / ions.INT.max()) * 10E4
    ions["MSF_INT"] = norm
    ## 2. Pick matched ions ##
    proof = proof[proof.PPM<=ftol]
    matched_ions = proof.join(ions.set_index('MZ'), lsuffix='_x', rsuffix='_y', how='left')
    if len(matched_ions) == 0:
        hs = 0
        return(hs)
    ## 3. Adjust intensity
    matched_ions.MSF_INT = matched_ions.MSF_INT / 10E2
    ## 4. Hyperscore ## # Consider modified ions but not charged ions? unclear
    matched_ions["SERIES"] = matched_ions.apply(lambda x: x.FRAGS[0], axis=1)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('+', '', regex=False)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('*', '', regex=False)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('#', '', regex=False)
    temp = matched_ions.copy()
    # TRY use only charge less than 2maybe that's why only 3 and 4 have extra ions found.
    # temp = temp.drop_duplicates(subset='FRAGS', keep="first") # Count each kind of fragment only once
    try:
        n_b = temp.SERIES.value_counts()['b']
        i_b = matched_ions[matched_ions.SERIES=='b'].MSF_INT.sum()
    except KeyError:
        n_b = 1 # So that hyperscore will not be 0 if one series is missing
        i_b = 1
    try:
        n_y = temp.SERIES.value_counts()['y']
        i_y = matched_ions[matched_ions.SERIES=='y'].MSF_INT.sum()
    except KeyError:
        n_y = 1
        i_y = 1
    try:
        hs = math.log10(math.factorial(n_b) * math.factorial(n_y) * i_b * i_y)
    except ValueError:
        hs = 0
    if hs < 0:
        hs = 0
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
    omod = []
    opos = []
    for pos, value in modlist:
        peptide = peptide[:pos] + '[' + str(value) + ']' + peptide[pos:]
        omod.append(value)
        opos.append(pos-1)
    return(peptide, omod, opos)

def getTheoMH(charge, sequence, mods, pos, nt, ct, mass):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    AAs = dict(mass._sections['Aminoacids'])
    MODs = dict(mass._sections['Fixed Modifications'])
    m_proton = mass.getfloat('Masses', 'm_proton')
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    total_aas = 2*m_hydrogen + m_oxygen
    total_aas += charge*m_proton
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
        if i in pos:
            total_aas += float(mods[pos.index(i)])
    MH = total_aas - (charge-1)*m_proton
    return(MH)

def expSpectrum(ions):
    '''
    Prepare experimental spectrum.
    '''        
    ions["ZERO"] = 0
    ions["CCU"] = ions.MZ - 0.01
    ions.reset_index(drop=True, inplace=True)
    
    bind = pd.DataFrame(list(itertools.chain.from_iterable(zip(list(ions['CCU']),list(ions['MZ'])))), columns=["MZ"])
    bind["REL_INT"] = list(itertools.chain.from_iterable(zip(list(ions['ZERO']),list(ions['INT']))))
    bind["ZERO"] = 0
    bind["CCU"] = bind.MZ + 0.01
    
    spec = pd.DataFrame(list(itertools.chain.from_iterable(zip(list(bind['MZ']),list(bind['CCU'])))), columns=["MZ"])
    spec["REL_INT"] = list(itertools.chain.from_iterable(zip(list(bind['REL_INT']),list(bind['ZERO']))))
    
    median_rel_int = statistics.median(ions.INT)
    std_rel_int = np.std(ions.INT, ddof = 1)
    ions["NORM_REL_INT"] = (ions.INT - median_rel_int) / std_rel_int
    ions["P_REL_INT"] = scipy.stats.norm.cdf(ions.NORM_REL_INT) #, 0, 1)
    normspec = ions.loc[ions.P_REL_INT>0.81]
    if len(ions) > 0 and len(normspec) > 0:
        spec_correction = max(ions.INT)/statistics.mean(normspec.INT)
    else: spec_correction = 0
    # spec["CORR_INT"] = spec.REL_INT*spec_correction
    # spec.loc[spec['CORR_INT'].idxmax()]['CORR_INT'] = max(spec.REL_INT)
    # spec["CORR_INT"] = spec.apply(lambda x: max(ions.INT)-13 if x["CORR_INT"]>max(ions.INT) else x["CORR_INT"], axis=1)
    return(spec, ions, spec_correction)

def theoSpectrum(seq, mods, pos, len_ions, mass, dm=0):
    '''
    Prepare theoretical fragment matrix.

    '''
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    ## Y SERIES ##
    #ipar = list(range(1,len(seq)))
    outy = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        yn = list(seq[i:])
        if i > 0: nt = False
        else: nt = True
        fragy = getTheoMH(0,yn,mods,pos,nt,True,mass) + dm # TODO only add +dm to fragments up until n_pos
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(0,bn,mods,pos,True,ct,mass) - 2*m_hydrogen - m_oxygen + dm # TODO only add +dm to fragments up until n_pos
        outb[i:] = fragb
    
    ## FRAGMENT MATRIX ##
    yions = outy.T
    bions = outb.iloc[::-1].T
    spec = pd.concat([bions, yions], axis=1)
    spec.columns = range(spec.columns.size)
    spec.reset_index(inplace=True, drop=True)
    return(spec)

def errorMatrix(mz, theo_spec, mass):
    '''
    Prepare ppm-error and experimental mass matrices.
    '''
    m_proton = mass.getfloat('Masses', 'm_proton')
    exp = pd.DataFrame(np.tile(pd.DataFrame(mz), (1, len(theo_spec.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 2 ##
    mzs2 = pd.DataFrame(mz)*2 - m_proton
    mzs2 = pd.DataFrame(np.tile(pd.DataFrame(mzs2), (1, len(exp.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 3 ##
    mzs3 = pd.DataFrame(mz)*3 - m_proton*2
    mzs3 = pd.DataFrame(np.tile(pd.DataFrame(mzs3), (1, len(exp.columns)))) 
    
    ## PPM ERRORS ##
    terrors = (((exp - theo_spec)/theo_spec)*1000000).abs()
    terrors2 =(((mzs2 - theo_spec)/theo_spec)*1000000).abs()
    terrors3 = (((mzs3 - theo_spec)/theo_spec)*1000000).abs()
    return(terrors, terrors2, terrors3, exp)

def makeFrags(seq_len):
    '''
    Name all fragments.
    '''
    frags = pd.DataFrame(np.nan, index=list(range(0,seq_len*2)),
                         columns=["by", "by2", "by3", "bydm", "bydm2", "bydm3"])
    frags.by = ["b" + str(i) for i in list(range(1,seq_len+1))] + ["y" + str(i) for i in list(range(1,seq_len+1))[::-1]]
    frags.by2 = frags.by + "++"
    frags.by3 = frags.by + "+++"
    frags.bydm = frags.by + "*"
    frags.bydm2 = frags.by + "*++"
    frags.bydm3 = frags.by + "*+++"
    return(frags)

def assignIons(theo_spec, dm_theo_spec, frags, dm, mass):
    m_proton = mass.getfloat('Masses', 'm_proton')
    assign = pd.concat([frags.by, theo_spec.iloc[0]], axis=1)
    assign.columns = ['FRAGS', '+']
    assign["++"] = (theo_spec.iloc[0]+m_proton)/2
    assign["+++"] = (theo_spec.iloc[0]+2*m_proton)/3
    assign["*"] = dm_theo_spec.iloc[0]
    assign["*++"] = (dm_theo_spec.iloc[0]+m_proton)/2
    c_assign = pd.DataFrame(list(assign["+"]) + list(assign["++"]) + list(assign["+++"]))
    c_assign = pd.concat([c_assign, pd.DataFrame(list(assign["*"])), pd.DataFrame(list(assign["*++"]))])
    c_assign.columns = ["MZ"]
    c_assign_frags = pd.DataFrame(list(frags.by) + list(frags.by + "++") + list(frags.by + "+++"))
    c_assign_frags = pd.concat([c_assign_frags, pd.DataFrame(list(frags.by + "*")), pd.DataFrame(list(frags.by + "*++"))])
    c_assign["FRAGS"] = c_assign_frags
    c_assign["ION"] = c_assign.apply(lambda x: re.findall(r'\d+', x.FRAGS)[0], axis=1)
    c_assign["CHARGE"] = c_assign.apply(lambda x: x.FRAGS.count('+'), axis=1).replace(0, 1)
    return(c_assign)

def makeAblines(texp, minv, assign, ions):
    masses = pd.concat([texp[0], minv], axis = 1)
    matches = masses[(masses < 51).sum(axis=1) >= 0.001]
    matches.reset_index(inplace=True, drop=True)
    if len(matches) <= 0:
        matches = pd.DataFrame([[1,3],[2,4]])
        proof = pd.DataFrame([[0,0,0,0]])
        proof.columns = ["MZ","FRAGS","PPM","INT"]
        return(proof)
    matches_ions = pd.DataFrame(list(itertools.product(list(range(0, len(matches))), list(range(0, len(assign))))))
    matches_ions.columns = ["mi", "ci"]
    matches_ions["temp_ci"] = list(assign.iloc[matches_ions.ci,0])
    matches_ions["temp_mi"] = list(matches.iloc[matches_ions.mi,0])
    matches_ions["temp_ci1"] = list(assign.iloc[matches_ions.ci,1])
    matches_ions["temp_mi1"] = list(matches.iloc[matches_ions.mi,1])
    matches_ions["check"] = abs(matches_ions.temp_mi-matches_ions.temp_ci)/matches_ions.temp_ci*1000000
    matches_ions = matches_ions[matches_ions.check<=51]
    matches_ions = matches_ions.drop(["mi", "ci", "temp_ci", "check"], axis = 1)
    matches_ions.columns = ["MZ","FRAGS","PPM"]
    if matches_ions.empty:
        proof = pd.DataFrame([[0,0,0,0]])
        proof.columns = ["MZ","FRAGS","PPM","INT"]
        return(proof, False)
    proof = matches_ions.set_index('MZ').join(ions[['MZ','INT']].set_index('MZ'), lsuffix='_x', rsuffix='_y', how='left', on='MZ')
    if len(proof)==0:
        mzcycle = itertools.cycle([ions.MZ.iloc[0], ions.MZ.iloc[1]])
        proof = pd.concat([matches_ions, pd.Series([next(mzcycle) for count in range(len(matches_ions))], name="INT")], axis=1)
    return(proof)

def findClosest(dm, dmdf, dmtol, pos):
    exp = pd.DataFrame(['EXPERIMENTAL', dm, [pos], 0]).T
    exp.columns = ['name', 'mass', 'site', 'distance']
    dmdf["distance"] = abs(dmdf.mass - dm)
    closest = dmdf[dmdf.distance<=dmtol]
    closest = pd.concat([closest, exp])
    closest.sort_values(by=['mass'], inplace=True, ascending=True)
    closest.reset_index(drop=True, inplace=True)
    return(closest)

def findPos(dm_set, plainseq):
    def _where(sites, plainseq):
        sites = sites.site
        subpos = []
        for s in sites:
            if s == 'Anywhere':
                subpos = list(range(0, len(plainseq)))
                break
            elif s == 'NM' or s == 'exp':
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

def miniVseq(sub, plainseq, mods, pos, mass, ftol, dmtol, dmdf,
             exp_spec, ions, spec_correction):
    ## DM ##
    exp_pos = 'exp'
    dm_set = findClosest(sub.DM, dmdf, dmtol, exp_pos) # Contains experimental DM
    dm_set = findPos(dm_set, plainseq)
    theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), mass)
    terrors, terrors2, terrors3, texp = errorMatrix(ions.MZ, theo_spec, mass)
    closest_proof = []
    closest_dm = []
    closest_name = []
    closest_pos = []
    for index, row in dm_set.iterrows():
        dm = row.mass
        for dm_pos in row.idx:
            ## DM OPERATIONS ##
            if dm_pos == -1:
                dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), mass, dm)
            else:
                mods.append(dm)
                pos.append(dm_pos)
                dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), mass)
            dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(ions.MZ, dm_theo_spec, mass)
            ## FRAGMENT NAMES ##
            frags = makeFrags(len(plainseq))
            dmterrors.columns = frags.by
            dmterrors2.columns = frags.by2
            dmterrors3.columns = frags.by3
            ## ASSIGN IONS WITHIN SPECTRA ##
            assign = assignIons(theo_spec, dm_theo_spec, frags, dm, mass)
            ## PPM ERRORS ##
            if sub.Charge == 2:
                ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
                if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, dmterrors, dmterrors2]).min(0))
            elif sub.Charge < 2:
                ppmfinal = pd.DataFrame(np.array([terrors]).min(0))
                if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, dmterrors]).min(0))
            elif sub.Charge >= 3:
                ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3]).min(0))
                if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3, dmterrors, dmterrors2, dmterrors3]).min(0))
            else:
                sys.exit('ERROR: Invalid charge value!')
            ppmfinal["minv"] = ppmfinal.apply(lambda x: x.min() , axis = 1)
            minv = ppmfinal["minv"]
            ## ABLINES ##
            proof = makeAblines(texp, minv, assign, ions)
            proof.INT = proof.INT * spec_correction
            proof.INT[proof.INT > max(exp_spec.REL_INT)] = max(exp_spec.REL_INT) - 3
            proof = proof[proof.PPM<=ftol]
            closest_proof.append(proof)
            closest_dm.append(dm)
            closest_name.append(row['name'])
            closest_pos.append(dm_pos)
    return(closest_proof, closest_dm, closest_name, closest_pos)

def parallelFragging(query, parlist):
    m_proton = 1.007276
    scan = query.scannum
    charge = query.charge
    MH = query.precursor_neutral_mass + (m_proton)
    plain_peptide = query.peptide
    if pd.isnull(query.modification_info): 
        sequence = plain_peptide
        mod = []
        pos = []
    else:
        sequence, mod, pos = insertMods(plain_peptide, query.modification_info)
    spectrum = query.spectrum
    dm = query.massdiff
    # TODO use calc neutral mass?
    # Make a Vseq-style query
    sub = pd.Series([scan, charge, MH, sequence, spectrum, dm],
                    index = ["FirstScan", "Charge", "MH", "Sequence", "Spectrum", "DM"])
    exp_spec, exp_ions, spec_correction = expSpectrum(sub.Spectrum)
    proof, dm, name, position = miniVseq(sub, plain_peptide, mod, pos,
                                         parlist[0], parlist[1], parlist[2],
                                         parlist[3], exp_spec, exp_ions, spec_correction)
    hyperscores = []
    check = []
    hss = []
    ufrags = []
    for i in list(range(0, len(dm))):
        total = sum(list(proof[i].index)) + proof[i].INT.sum() # proof[i].MZ.sum(), MZ is now index
        if total in check:
            hscore = hss[check.index(total)]
            frags = ufrags[check.index(total)]
        else:
            hscore = hyperscore(exp_ions, proof[i], parlist[2])
            proof[i].FRAGS = proof[i].FRAGS.str.replace('+', '')
            proof[i].FRAGS = proof[i].FRAGS.str.replace('*', '')
            frags = proof[i].FRAGS.nunique()
            check = check + [total]
            hss = hss + [hscore]
            ufrags = ufrags + [frags]
        hyperscores = hyperscores + [[name[i], dm[i], position[i], frags, hscore]]
    hyperscores = pd.DataFrame(hyperscores, columns = ['name', 'dm', 'site', 'matched_ions', 'hyperscore'])
    best = hyperscores[hyperscores.hyperscore==hyperscores.hyperscore.max()]
    best.sort_values(by=['matched_ions'], inplace=True, ascending=True) #TODO In case of tie (also prefer theoretical rather than experimental)
    best.reset_index(drop=True, inplace=True)
    best = best.head(1)
    exp = hyperscores[hyperscores['name']=='EXPERIMENTAL']
    return([MH, best.dm[0], sequence, best.matched_ions[0], best.hyperscore[0], best['name'][0],
            int(exp.matched_ions), float(exp.hyperscore), best.site[0]])

def main(args):
    '''
    Main function
    '''
    # Parameters
    chunks = int(mass._sections['Parameters']['batch_size'])
    ftol = float(mass._sections['Parameters']['f_tol'])
    dmtol = float(mass._sections['Parameters']['dm_tol'])
    # Read results file from MSFragger
    logging.info("Reading MSFragger file...")
    df = pd.read_csv(Path(args.infile), sep="\t")
    logging.info("\t" + str(len(df)) + " lines read.")
    # Read raw file
    msdata, mode, index2 = readRaw(Path(args.rawfile))
    # Read DM file
    logging.info("Reading DM file...")
    dmdf = pd.read_csv(Path(args.dmfile), sep="\t")
    dmdf.columns = ["name", "mass", "site"]
    dmdf.site = dmdf.site.apply(literal_eval)
    dmdf.site = dmdf.apply(lambda x: list(dict.fromkeys(x.site)), axis=1)
    logging.info("\t" + str(len(dmdf)) + " theoretical DMs read.")
    # Prepare to parallelize
    logging.info("Refragging...")
    df["spectrum"] = df.apply(lambda x: locateScan(x.scannum, mode, msdata, index2), axis=1)
    indices, rowSeries = zip(*df.iterrows())
    rowSeries = list(rowSeries)
    tqdm.pandas(position=0, leave=True)
    if len(df) <= chunks:
        chunks = math.ceil(len(df)/args.n_workers)
    parlist = [mass, ftol, dmtol, dmdf]
    logging.info("\tBatch size: " + str(chunks) + " (" + str(math.ceil(len(df)/chunks)) + " batches)")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        refrags = list(tqdm(executor.map(parallelFragging,
                                         rowSeries,
                                         itertools.repeat(parlist),
                                         chunksize=chunks),
                            total=len(rowSeries)))
    df = df.drop('spectrum', axis = 1)
    df['templist'] = refrags
    df['REFRAG_MH'] = pd.DataFrame(df.templist.tolist()).iloc[:, 0]. tolist()
    df['REFRAG_exp_DM'] = pd.DataFrame(df.templist.tolist()).iloc[:, 6]. tolist()
    df['REFRAG_exp_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 7]. tolist()
    df['REFRAG_DM'] = pd.DataFrame(df.templist.tolist()).iloc[:, 1]. tolist()
    df['REFRAG_site'] = pd.DataFrame(df.templist.tolist()).iloc[:, 8]. tolist()
    df['REFRAG_sequence'] = pd.DataFrame(df.templist.tolist()).iloc[:, 2]. tolist()
    df['REFRAG_ions_matched'] = pd.DataFrame(df.templist.tolist()).iloc[:, 3]. tolist()
    df['REFRAG_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 4]. tolist()
    df['REFRAG_name'] = pd.DataFrame(df.templist.tolist()).iloc[:, 5]. tolist()
    df = df.drop('templist', axis = 1)
    try:
        refragged = len(df)-df.REFRAG_name.value_counts()['EXPERIMENTAL']
    except KeyError:
        refragged = len(df)
    prefragged = round((refragged/len(df))*100,2)
    logging.info("\t" + str(refragged) + " (" + str(prefragged) + "%) refragged PSMs.")
    logging.info("Writing output file...")
    outpath = Path(os.path.splitext(args.infile)[0] + "_REFRAG.tsv")
    df.to_csv(outpath, index=False, sep='\t', encoding='utf-8')
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
    
    parser.add_argument('-i',  '--infile', required=True, help='MSFragger results file')
    parser.add_argument('-r',  '--rawfile', required=True, help='MS Data file (MGF or MZML)')
    parser.add_argument('-d',  '--dmfile', required=True, help='DeltaMass file')
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
    log_file = args.infile[:-4] + 'ReFrag_log.txt'
    log_file_debug = args.infile[:-4] + 'ReFrag_log_debug.txt'
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