# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:33:27 2023

@author: alaguillog
"""

###############################################################################
### STEP 1: IMPORT MODULES AND DEFINE FUNCTIONS ###############################
import matplotlib.pyplot as plt
import pandas as pd
import re

def geneName(prot):
    try:
        # gene = prot.split("=")[3].split(" ")[0]
        gene = re.findall('GN=[^ ]+', prot)[0]
        gene = gene.replace('GN=', '')
    except:
        gene = "UNKNOWN"
    return(gene)
###############################################################################
###############################################################################

###############################################################################
### STEP 2: DEFINE PARAMETERS #################################################
# Define non-modified deltamass range:
# (deltamasses between mindm and maxdm will be considered as non-modified)
mindm = -3
maxdm = 0.8
# Define input file (path to DisXtractor output)
file = r"C:\All_FDR_0.05.tsv"
# Define output files to be created
prot_out = r"C:\report_protein.tsv" # Peptide-level report
pep_out = r"C:\report_peptide.tsv" # Protein-level report
unk_out = r"C:\report_unknown.tsv" # 
# Define path to theoretical deltamass table (unimod, etc...)
unimod = r"C:\unimod.tsv" # A default table is in the ReFrag repository
###############################################################################
###############################################################################

###############################################################################
### STEP 3: READ AND PROCESS INPUT FILES ######################################
# Read theoretical deltamass table:
unimod = pd.read_csv(unimod, sep='\t')
# Read DisXtractor output:
df = pd.read_csv(file, sep='\t')
# Mark as non-modified all deltamasses between mindm and maxdm:
cleanup = df[(df.REFRAG_DM>=mindm)&(df.REFRAG_DM<maxdm)].index
df.loc[df.index.isin(cleanup), 'REFRAG_DM'] = 0
df.loc[df.index.isin(cleanup), 'REFRAG_name'] = 'Non-modified'
###############################################################################
###############################################################################

###############################################################################
### OPTIONAL STEP: PLOT DELTAMASS DISTRIBUTION ################################
fig, axs = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
plt.hist(df.REFRAG_DM, bins=730) # Adjust number of bins if necessary
plt.title("DeltaMass Distribution")
plt.xlabel("DeltaMass")
plt.ylabel("Frequency")
###############################################################################
###############################################################################

###############################################################################
### STEP 4: GENERATE PROTEIN-LEVEL REPORT #####################################
# You should not need to change anything in this step
results = []
for prot, gprot in df.groupby("protein"):
    for band, gband in gprot.groupby("File"):
        nm = gband[gband.REFRAG_name=='Non-modified']
        psm_nm = len(nm)
        psm_all = len(gband)
        sp_nm = nm.REFRAG_sp_score.sum()
        for mod, gmod in gband.groupby("REFRAG_name"):
            if mod != "EXPERIMENTAL" and mod != "Non-modified":
                sp_mod = gmod.REFRAG_sp_score.sum()
                results.append([prot, band, mod, psm_nm, psm_all, sp_nm, sp_mod])
results = pd.DataFrame(results)
results.columns = ["protein", "band", "mod", "PSMs_NM", "PSMs_ALL", "sum_spscore_NM", "sum_spscore_MOD"]
cols = ["protein", "band", "experiment", "PSMs_NM", "PSMs_ALL", "sum_spscore_NM"] + list(results["mod"].unique())
df_result = pd.DataFrame(index=pd.RangeIndex(0, len(results.groupby(["protein", "band"]))), columns=cols)
final_results = []
counter = 0
for i in results.groupby(["protein", "band"]):
    modlist = list(i[1]["mod"])
    modsp = list(i[1]["sum_spscore_MOD"])
    df_result.iloc[counter][["protein", "band", "PSMs_NM", "PSMs_ALL", "sum_spscore_NM"]] = [i[1].protein.iloc[0],
                              i[1].band.iloc[0],
                              i[1].PSMs_NM.iloc[0],
                              i[1].PSMs_ALL.iloc[0],
                              i[1].sum_spscore_NM.iloc[0]]
    c = 0
    for j in list(i[1]["mod"]):
        df_result.iloc[counter][j] = list(i[1]["sum_spscore_MOD"])[c]
        c += 1
    counter += 1
    print(counter)
df_result = df_result.fillna(0)
df_result.insert(1, "gene", 0)
df_result.gene = df_result.apply(lambda x: geneName(x.protein), axis=1)
fixcols = [list(unimod[unimod["Full Name"]==m].Title)[0] + ' (' + str(float(list([unimod[unimod["Full Name"]==m].mono_mass])[0])) + ')' if m in list(unimod["Full Name"]) else m for m in df_result.columns]
df_result.columns = fixcols
df_result = df_result.copy()
df_result.insert(7, "sum_spscore_MOD", 0)
df_result.sum_spscore_MOD = df_result.iloc[:,8:].T.sum()
df_result.sum_spscore_MOD = df_result.sum_spscore_MOD - df_result['Oxidation (15.994915)'] - df_result['Methyl (14.01565)']
df_result.insert(7, "sum_spscore_NM+Oxi+Met", 0)
df_result['sum_spscore_NM+Oxi+Met'] = df_result.sum_spscore_NM + df_result['Oxidation (15.994915)'] + df_result['Methyl (14.01565)']
df_result.insert(9, "sum_spscore_ALL", 0)
df_result.sum_spscore_ALL = df_result["sum_spscore_NM+Oxi+Met"] + df_result.sum_spscore_MOD
df_result["experiment"] = df_result.band
###############################################################################
###############################################################################

###############################################################################
### STEP 5: ADD EXPERIMENT AND BAND COLUMNS TO REPORT #########################
# These columns are generated from the file names, so the exact method to
# generate them is experiment-specific. Examples are provided here for
# DIA experiments we have already analyzed:

# olsen
df_result.experiment = df_result.experiment.str.replace("_01|_02","")
df_result.band = df_result.band.str.replace("20230802_AST_Neo1_KBE_15min_DIA_Phos_25ug_2th2p5ms_0","")
df_result.band = df_result.band.astype(int)

df_result.experiment = "25052023_HeK_200ng_Tritation-180SPD"
df_result.band = df_result.band.str.replace("25052023_HeK_200ng_Tritation-180SPD_0|_20230525155651|_20230525160533|_20230525161412","")
df_result.band = df_result.band.astype(int)

# human muscle mitoch
df_result.experiment = df_result.experiment.str.replace("1|2|3|4|5|6|7|8|9|0","")
df_result.band = df_result.band.str.replace("MR|FR|MEX|FEX","")
df_result.band = df_result.band.astype(int)

# complexom_macrofagos
df_result.experiment = df_result.experiment.str.replace("PAM2","Pam")
df_result.experiment = df_result.experiment.str.replace("ZYM","Zym")
df_result.experiment = df_result.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_","")
df_result.band = df_result.band.str.replace("Control|PAM2|Zym|ZYM|D|Pam|_", "")
df_result.band = df_result.band.astype(int)

# DKO
df_result.experiment = df_result.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_DIS","")
df_result.band = df_result.band.str.replace("[^0-9]*", "")
df_result.band = df_result.band.astype(int)

# mutante cox7a2
df_result["tissue"] = df_result.band
df_result.tissue = df_result.tissue.str.replace("1|2|3|4|5|6|7|8|9|0|_|WT_1|KO_1|WT_2|KO_2|WT|KO", "")
df_result["111/113"] = df_result.band
df_result["111/113"] = df_result["111/113"].str.replace("Heart|Liver|_WT_1|_KO_1|_WT_2|_KO_2|_WT|_KO", "")
df_result["111/113"] = df_result["111/113"].str.replace("_113", "B")
df_result["111/113"] = df_result["111/113"].str.replace("_111", "A")
df_result["111/113"] = df_result["111/113"].str.replace("1|2|3|4|5|6|7|8|9|0|_","")
df_result["111/113"] = df_result["111/113"].str.replace("B", "113")
df_result["111/113"] = df_result["111/113"].str.replace("A", "111")
df_result.experiment = df_result.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_|Heart|Liver","")
df_result.band = df_result.band.str.replace("113|111|WT_1|WT_2|KO_1|KO_2|WT|KO|Heart|Liver|_","")
df_result.band = df_result.band.astype(int)

# serumLF
df_result.experiment = "JAL_Pl_dis"
df_result.band = df_result.band.str.replace("JAL_Pl_dis","")
df_result.band = df_result.band.astype(int)

# supermito
df_result.experiment = df_result.experiment.str.split('_').str[:-2].str.join("_")
df_result.experiment = df_result.experiment.str.replace("Liver_BL6", "BL6_Liver")
df_result.experiment = df_result.experiment.str.replace("^Brain$", "Brain_CD1")
df_result.experiment = df_result.experiment.str.replace("^Heart$", "Heart_CD1")
df_result.experiment = df_result.experiment.str.replace("^Liver$", "Liver_CD1")
df_result.band = df_result.band.str.split('_').str[-2]
df_result.band = df_result.band.astype(int)

# C4
df_result.experiment = df_result.experiment.str.replace("1|2|3|4|5|6|7|8|9|0","")
df_result.experiment = df_result.experiment.str.replace("_$","")
df_result.band = df_result.band.str.replace("ZF_DKO_|ZF_WT_|ZF_S_|ZF_AI_","")
df_result.band = df_result.band.astype(int)
    # cleanup duplicates from trembl #
clean_df = []
for i, j in df_result.groupby(["experiment", "band", "gene"]):
    if len(j) > 1:
        if j.protein.str.startswith('sp').sum() > 0:
            protein = j[j.protein.str.startswith('sp')].protein.iloc[0]
            temp = pd.Series([protein, i[2], i[1], i[0]])
            temp.index = ['protein', 'gene', 'band', 'experiment']
            temp = pd.concat([temp, j.iloc[:,4:].sum()])
            clean_df.append(temp)
        else:
            protein = j.protein.iloc[0]
            temp = pd.Series([protein, i[2], i[1], i[0]])
            temp.index = ['protein', 'gene', 'band', 'experiment']
            temp = pd.concat([temp, j.iloc[:,4:].sum()])
            clean_df.append(temp)
    else:
        indices, rowSeries = zip(*j.iterrows())
        for k in rowSeries:
            clean_df.append(k)
df_result = pd.concat(clean_df,axis=1).T

# NADIA ZF
df_result.experiment = df_result.experiment.str.replace("_1|_2|_3|_4|_5|_6|_7|_8|_9|1|2|3|4|5|6|7|8|9|0","")
df_result.band = df_result.band.str.replace("ZF_KO_NHF_|ZF_WT_NHF_","")
df_result.band = df_result.band.astype(int)
    # cleanup duplicates from trembl #
clean_df = []
for i, j in df_result.groupby(["experiment", "band", "gene"]):
    if len(j) > 1:
        if j.protein.str.startswith('sp').sum() > 0:
            protein = j[j.protein.str.startswith('sp')].protein.iloc[0]
            temp = pd.Series([protein, i[2], i[1], i[0]])
            temp.index = ['protein', 'gene', 'band', 'experiment']
            temp = pd.concat([temp, j.iloc[:,4:].sum()])
            clean_df.append(temp)
        else:
            protein = j.protein.iloc[0]
            temp = pd.Series([protein, i[2], i[1], i[0]])
            temp.index = ['protein', 'gene', 'band', 'experiment']
            temp = pd.concat([temp, j.iloc[:,4:].sum()])
            clean_df.append(temp)
    else:
        indices, rowSeries = zip(*j.iterrows())
        for k in rowSeries:
            clean_df.append(k)
df_result = pd.concat(clean_df,axis=1).T
###############################################################################
###############################################################################

###############################################################################
### STEP 6: WRITE PROTEIN-LEVEL REPORT TO A FILE ##############################
df_result.to_csv(prot_out, sep="\t", index=False)
df[(df.REFRAG_name=="EXPERIMENTAL")&(df.REFRAG_DM>0)].to_csv(unk_out,
                                                             sep="\t",
                                                             index=False)
###############################################################################
###############################################################################

###############################################################################
### STEP 7: GENERATE PEPTIDE-LEVEL REPORT #####################################
# You should not need to change anything in this step
def annotatePos(seq, pos, dm, name):
    if name not in ['EXPERIMENTAL', 'Non-modified']:
        pos = int(pos[1:])
        a = seq[:pos+1] + '[' + str(dm) + ']' + seq[pos+1:]
    else:
        a = seq
    return(a)
df["modpep"] = df.apply(lambda x: annotatePos(x.peptide, x.REFRAG_site, x.REFRAG_DM, x.REFRAG_name), axis=1)
pepdf = df.copy()
pepdf = pepdf[pepdf.REFRAG_name!='EXPERIMENTAL']
results = []
for pep, gpep in pepdf.groupby("modpep"):
    for band, gband in gpep.groupby("File"):
        results.append([pep, band, list(gband.REFRAG_name.unique())[0], list(gband.REFRAG_DM.unique())[0], list(gband.protein.unique())[0], len(gband), gband.REFRAG_sp_score.sum()])
results = pd.DataFrame(results)
results.columns = ["peptide", "band", "mod", "DM", "protein", "PSMs", "sum_spscore"]
results.insert(4, "gene", 0)
results["experiment"] = results.band
###############################################################################
###############################################################################

###############################################################################
### STEP 8: ADD EXPERIMENT AND BAND COLUMNS TO REPORT #########################
# These columns are generated from the file names, so the exact method to
# generate them is experiment-specific. Examples are provided here for
# DIA experiments we have already analyzed:
    
# olsen
results.experiment = results.experiment.str.replace("_01|_02","")
results.band = results.band.str.replace("20230802_AST_Neo1_KBE_15min_DIA_Phos_25ug_2th2p5ms_0","")
results.band = results.band.astype(int)

results.experiment = "25052023_HeK_200ng_Tritation-180SPD"
results.band = results.band.str.replace("25052023_HeK_200ng_Tritation-180SPD_0|_20230525155651|_20230525160533|_20230525161412","")
results.band = results.band.astype(int)

# human muscle mitoch
results.experiment = results.experiment.str.replace("1|2|3|4|5|6|7|8|9|0","")
results.band = results.band.str.replace("MR|FR|MEX|FEX","")
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]

# complexom macrofagos
results.experiment = results.experiment.str.replace("PAM2","Pam")
results.experiment = results.experiment.str.replace("ZYM","Zym")
results.experiment = results.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_","")
results.band = results.band.str.replace("Control|PAM2|Zym|ZYM|D|Pam|_", "")
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]

# DKO
results.experiment = results.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_DIS","")
results.band = results.band.str.replace("[^0-9]*", "")
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]

# mutante cox7a2
results["tissue"] = results.band
results.tissue = results.tissue.str.replace("1|2|3|4|5|6|7|8|9|0|_|WT_1|WT_2|KO_1|KO_2|WT|KO", "")
results["111/113"] = results.band
results["111/113"] = results["111/113"].str.replace("Heart|Liver|_WT_1|_KO_1|_WT_2|_KO_2|_WT|_KO", "")
results["111/113"] = results["111/113"].str.replace("_113", "B")
results["111/113"] = results["111/113"].str.replace("_111", "A")
results["111/113"] = results["111/113"].str.replace("1|2|3|4|5|6|7|8|9|0|_","")
results["111/113"] = results["111/113"].str.replace("B", "113")
results["111/113"] = results["111/113"].str.replace("A", "111")
results.experiment = results.experiment.str.replace("1|2|3|4|5|6|7|8|9|0|_|Heart|Liver","")
results.band = results.band.str.replace("113|111|WT_1|WT_2|KO_1|KO_2|WT|KO|Heart|Liver|_","")
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
# results.band = results.band.str.replace("551|552","55")
# results.band = results.band.str.replace("101|102","10")
# results.band = results.band.str.replace("161|162","16")
# results.band = results.band.str.replace("131|132","13")
# results.band = results.band.str.replace("41|42","4")
# results.band = results.band.str.replace("51|52","5")
# results.band = results.band.str.replace("61|62","6")
# results.band = results.band.str.replace("71|72","7")
# results.band = results.band.str.replace("81|82","8")
results.band = results.band.astype(int)
results = results[['peptide', 'tissue', 'band', '111/113', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]

# serum
results.experiment = "JAL_Pl_dis"
results.band = results.band.str.replace("JAL_Pl_dis","")
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.band = results.band.astype(int)

# supermito
results.experiment = results.experiment.str.split('_').str[:-2].str.join("_")
results.experiment = results.experiment.str.replace("Liver_BL6", "BL6_Liver")
results.experiment = results.experiment.str.replace("^Brain$", "Brain_CD1")
results.experiment = results.experiment.str.replace("^Heart$", "Heart_CD1")
results.experiment = results.experiment.str.replace("^Liver$", "Liver_CD1")
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.band = results.band.str.split('_').str[-2]
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]

# C4
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.experiment = results.experiment.str.replace("1|2|3|4|5|6|7|8|9|0","")
results.experiment = results.experiment.str.replace("_$","")
results.band = results.band.str.replace("ZF_DKO_|ZF_WT_|ZF_S_|ZF_AI_","")
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]


# NADIA ZF
results.gene = results.apply(lambda x: geneName(x.protein), axis=1)
results.experiment = results.experiment.str.replace("_1|_2|_3|_4|_5|_6|_7|_8|_9|1|2|3|4|5|6|7|8|9|0","")
results.band = results.band.str.replace("ZF_KO_NHF_|ZF_WT_NHF_","")
results.band = results.band.astype(int)
results = results[['peptide', 'band', 'experiment', 'mod', 'DM', 'protein', 'gene', 'PSMs', 'sum_spscore']]
###############################################################################
###############################################################################

###############################################################################
### STEP 9: WRITE PEPTIDE-LEVEL REPORT TO A FILE ##############################
results.to_csv(pep_out, sep="\t", index=False)
###############################################################################
###############################################################################



