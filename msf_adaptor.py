# Updated 14-Oct-2024 to modify assignation of modification site when several aminoacids are suggested by MSF. If more than 1 site is posible, choose the center of the aa distribution.

import os
import re
import argparse
import pandas as pd
import numpy as np
import math
import logging
from re import findall as refindall

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(
    description='Adapt the MSFragger result',
    epilog='''
    Example:
        python msf_adaptor.py -i MSFragger result

    ''')

parser.add_argument('-i',  '--input', required=True, help='Input')
args = parser.parse_args()

# Function to find positions
def msf_pos(x):
    if re.search('[a-z]', x):
        pos = [i for i, c in enumerate(x) if c.islower()]
        aa = [x[p].upper() for p in pos]
        pos2 = [f"{a}{p}" for a, p in zip(aa, pos)]
        pos3 = ";".join(pos2)
    else:
        pos = len(x)
        pos3 = f"_{pos}"
    return pos3


# Get input file
input_file = args.input
logging.info(f"Input file: {input_file}")

# Get the directory and base name of the input file
input_dir = os.path.dirname(input_file)
input_base = os.path.splitext(os.path.basename(input_file))[0]

# Construct output file path with "_pos" appended to the filename
output_file = os.path.join(input_dir, f"{input_base}_pos.tsv")
logging.info(f"Output file will be saved as: {output_file}")

# Process file
logging.info("Reading input file...")
input_df = pd.read_csv(input_file, sep='\t')
logging.info(f"File {input_file} loaded successfully with {len(input_df)} rows.")

logging.info("Calculating precursor_MH and precursor_MZ...")
input_df['precursor_MH'] = (input_df.precursor_neutral_mass + 1.007276)
input_df['precursor_MZ'] = (input_df.precursor_MH + (input_df.charge-1)*1.007276) / input_df.charge

logging.info("Filling 'best_locs' where NaN values are present...")
input_df['best_locs'] = input_df.apply(lambda row: row['peptide'] if pd.isna(row['best_locs']) else row['best_locs'], axis=1)

logging.info("Applying msf_pos function to find modification sites...")
input_df['m_MSF_positions'] = input_df['best_locs'].apply(msf_pos)

logging.info("Selecting the modification site (center of aa distribution if multiple)...")
input_df['m_MSF'] = input_df.apply(lambda row: row['m_MSF_positions'].split(";")[math.ceil(len(row['m_MSF_positions'].split(";"))/2)-1] if len(row['m_MSF_positions'].split(";"))>1 else row['m_MSF_positions'], axis=1)
input_df['m_MSF'] = input_df['m_MSF'].str[1:].astype(int) + 1

logging.info("Calculating left and right positions of modification...")
input_df['m_MSF_left'] = input_df.apply(lambda row: len(row['m_MSF_positions'].split(";"))-(len(row['m_MSF_positions'].split(";"))-math.ceil(len(row['m_MSF_positions'].split(";"))/2))-1 if len(row['m_MSF_positions'].split(";"))>1 else 0, axis=1)
input_df['m_MSF_right'] = input_df.apply(lambda row: len(row['m_MSF_positions'].split(";"))-math.ceil(len(row['m_MSF_positions'].split(";"))/2) if len(row['m_MSF_positions'].split(";"))>1 else 0, axis=1)

logging.info("Updating 'delta_peptide' column based on positions...")
input_df['delta_peptide'] = input_df['peptide']

# Vectorized update of delta_peptide column
logging.info("Updating peptides where modifications involve mass differences...")
mask = input_df['m_MSF_positions'].str.contains('_')
input_df.loc[mask, 'delta_peptide'] = input_df.loc[mask, 'peptide'] + '_' + input_df.loc[mask, 'massdiff'].astype(str)
non_mask = ~mask
input_df.loc[non_mask, 'delta_peptide'] = input_df.loc[non_mask].apply(
    lambda row: row['delta_peptide'][:row['m_MSF']] + f"[{row['massdiff']}]" + row['delta_peptide'][row['m_MSF']:], axis=1
)

# Save the output
logging.info("Saving the updated DataFrame to output file...")
input_df.to_csv(output_file, sep='\t', index=False, quoting=False)