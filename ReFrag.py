# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:52 2022

@author: alaguillog
"""

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
        index1 = fr_ns.to_numpy() == 'SCANS='+str(int(scan))
        index1 = np.where(index1)[0][0]
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
        s = fr_ns.getSpectrum(scan-1)
        ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
        ions.columns = ["MZ", "INT"]
    return(ions)

def hyperscore(ions, proof): # TODO play with number of ions
    ## 1. Normalize intensity to 10^5
    norm = (ions.INT / ions.INT.max()) * 10E4
    ions["MSF_INT"] = norm
    ## 2. Pick matched ions ##
    matched_ions = pd.merge(proof, ions, on="MZ")
    ## 3. Adjust intensity
    matched_ions.MSF_INT = matched_ions.MSF_INT / 10E2
    ## 4. Hyperscore ##
    matched_ions["SERIES"] = matched_ions.apply(lambda x: x.FRAGS[0], axis=1)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('+', '')
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('*', '')
    temp = matched_ions.copy()
    temp.drop_duplicates(subset='FRAGS', keep="first")
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
    ions.reset_index(drop=True)
    
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

def theoSpectrum(seq, mods, pos, len_ions, dm, mass):
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
        fragy = getTheoMH(0,yn,mods,pos,nt,True,mass) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(0,bn,mods,pos,True,ct,mass) - 2*m_hydrogen - m_oxygen + dm
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
    proof = pd.merge(matches_ions, ions[['MZ','INT']], how="left", on="MZ")
    if len(proof)==0:
        mzcycle = itertools.cycle([ions.MZ.iloc[0], ions.MZ.iloc[1]])
        proof = pd.concat([matches_ions, pd.Series([next(mzcycle) for count in range(len(matches_ions))], name="INT")], axis=1)
    return(proof)

def findClosest(dm, dmdf, dmtol):
    exp = pd.DataFrame(['EXPERIMENTAL', dm, 0]).T
    exp.columns = ['name', 'mass', 'distance']
    dmdf["distance"] = abs(dmdf.mass - dm)
    closest = dmdf[dmdf.distance<=dmtol]
    closest = pd.concat([closest, exp])
    closest.sort_values(by=['mass'], inplace=True, ascending=True)
    closest.reset_index(drop=True, inplace=True)
    return(closest)

def miniVseq(sub, plainseq, mods, pos, mass, ftol, dmtol, dmdf):
    # TODO retry for every position in sequence as well
    ## DM ##
    dm_set = findClosest(sub.DM, dmdf, dmtol) # Contains experimental DM
    exp_spec, ions, spec_correction = expSpectrum(sub.Spectrum)
    theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), 0, mass)
    terrors, terrors2, terrors3, texp = errorMatrix(ions.MZ, theo_spec, mass)
    closest_ions = []
    closest_proof = []
    closest_dm = []
    closest_name = []
    for index, row in dm_set.iterrows():
        dm = row.mass
        ## DM OPERATIONS ##
        dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), dm, mass)
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
        closest_ions.append(ions)
        closest_proof.append(proof)
        closest_dm.append(dm)
        closest_name.append(row['name'])
    return(closest_ions, closest_proof, closest_dm, closest_name)

def parallelFragging(query, parlist):
    m_proton = 1.007276
    scan = query.scannum
    charge = query.charge
    MH = query.precursor_neutral_mass + (m_proton)
    plain_peptide = query.peptide
    sequence, mod, pos = insertMods(plain_peptide, query.modification_info)
    spectrum = query.spectrum
    dm = query.massdiff
    # TODO use calc neutral mass?
    # Make a Vseq-style query
    sub = pd.Series([scan, charge, MH, sequence, spectrum, dm],
                    index = ["FirstScan", "Charge", "MH", "Sequence", "Spectrum", "DM"])
    ions, proof, dm, name = miniVseq(sub, plain_peptide, mod, pos,
                                     parlist[0], parlist[1], parlist[2],
                                     parlist[3])
    hyperscores = pd.DataFrame(columns=['name', 'dm', 'matched_ions', 'hyperscore'])
    for i in list(range(0, len(dm))):
        hscore = hyperscore(ions[i], proof[i])
        proof[i].FRAGS = proof[i].FRAGS.str.replace('+', '')
        proof[i].FRAGS = proof[i].FRAGS.str.replace('*', '')
        candidate = pd.DataFrame([name[i], dm[i], proof[i].FRAGS.nunique(), hscore]).T
        candidate.columns = ['name', 'dm', 'matched_ions', 'hyperscore']
        hyperscores = pd.concat([hyperscores, candidate])
    best = hyperscores[hyperscores.hyperscore==hyperscores.hyperscore.max()]
    best.sort_values(by=['matched_ions'], inplace=True, ascending=True) # In case of tie
    best.reset_index(drop=True, inplace=True)
    best = best.head(1)
    return([MH, best.dm[0], sequence, best.matched_ions[0], best.hyperscore[0], best['name'][0]])

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
    dmdf.columns = ["name", "mass"]
    logging.info("\t" + str(len(dmdf)) + " theoretical DMs read.")
    # Prepare to parallelize
    df["spectrum"] = df.apply(lambda x: locateScan(x.scannum, mode, msdata, index2), axis=1)
    indices, rowSeries = zip(*df.iterrows())
    rowSeries = list(rowSeries)
    tqdm.pandas(position=0, leave=True)
    if len(df) <= chunks:
        chunks = math.ceil(len(df)/args.n_workers)
    parlist = [mass, ftol, dmtol, dmdf]
    logging.info("Refragging...")
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
    df['REFRAG_DM'] = pd.DataFrame(df.templist.tolist()).iloc[:, 1]. tolist()
    df['REFRAG_sequence'] = pd.DataFrame(df.templist.tolist()).iloc[:, 2]. tolist()
    df['REFRAG_ions_matched'] = pd.DataFrame(df.templist.tolist()).iloc[:, 3]. tolist()
    df['REFRAG_hyperscore'] = pd.DataFrame(df.templist.tolist()).iloc[:, 4]. tolist()
    df['REFRAG_name'] = pd.DataFrame(df.templist.tolist()).iloc[:, 5]. tolist()
    df = df.drop('templist', axis = 1)
    refragged = len(df)-df.REFRAG_name.value_counts()['EXPERIMENTAL']
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