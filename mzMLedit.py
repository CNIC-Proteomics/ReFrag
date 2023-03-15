# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:50:15 2023

@author: alaguillog
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

def prettyPrint(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        prettyPrint(node, current, i, depth+1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('  ' * depth)
        else:
            parent[index - 1].tail = '\n' + ('  ' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('  ' * (depth - 1))
            
def mzadjust(mz, charge):
    if charge == 2:
        mz = mz + (0.0004*mz + 0.107)
    # elif charge == 3:
        
    # elif charge == 4:
        
    return(mz)

def mzedit(tree, charge, first, adjust):  
    # for child in root:
    #     print(child.tag, child.attrib)
    # precursor = list(elem.iter('{http://psi.hupo.org/ms/mzml}precursor'))
    tree_list = list(tree.iter())
    # in_mz = 0
    accession = 0
    for n,i in enumerate(tree_list):
        if i.tag == '{http://psi.hupo.org/ms/mzml}selectedIon':
            # in_mz = 1
            # Add charge
            if first == 0:
                accession = tree_list[n+1].attrib['accession']
                chdict =  {
                           "cvRef": "MS",
                           "accession": str(accession),
                           "value": str(charge),
                           "name" : "charge state"
                          }
                charge_elem = ET.Element('{http://psi.hupo.org/ms/mzml}cvParam', chdict)
                charge_elem.tail = i.tail
                i.insert(1, charge_elem)
            # else: # TODO check
            #     tree_list[n+1].set("value", str(charge))
        if ((i.tag == '{http://psi.hupo.org/ms/mzml}cvParam') and (i.attrib['name'] == 'charge state')):
            i.set("value", str(charge))
        if adjust:
            if 'baseMZ' not in i.attrib:
                i.attrib["baseMZ"] = str(i.attrib['value'])
            if ((i.tag == '{http://psi.hupo.org/ms/mzml}cvParam') and (i.attrib['name'] == 'selected ion m/z')):
                # Modify mz
                new_mz = str(mzadjust(float(i.attrib["baseMZ"]), charge))
                i.set("value", new_mz)
            if ((i.tag == '{http://psi.hupo.org/ms/mzml}cvParam') and (i.attrib['name'] == 'isolation window target m/z')):
                # Modify mz # TODO isolation window target mz?
                new_mz = str(mzadjust(float(i.attrib["baseMZ"]), charge))
                i.set("value", new_mz)
                # in_mz = 0
    return(tree)

def main(args):
    '''
    Main function
    '''
    adjust = args.adjust
    
    # Make results directory
    if not os.path.exists(os.path.dirname(args.infile) + '\\mzMLedit'):
        os.mkdir(Path(os.path.dirname(args.infile) + '\\mzMLedit'))
    outdir = Path(os.path.dirname(args.infile) + '\\mzMLedit')
    
    # Read mzML file
    if '*' in args.infile: # wildcard
        infiles = glob.glob(args.infile)
        if len(infiles) == 0:
            sys.exit("ERROR: No files found matching pattern " + str(args.infile))
    else:
        infiles = [Path(args.infile)]
        
    for infile in infiles:
        logging.info("Reading mzML file (" + str(os.path.basename(infile)) + ")...")
        ET.register_namespace('', "http://psi.hupo.org/ms/mzml")
        tree = ET.parse(infile)
        logging.info("\t" + str(len(list(tree.iter('{http://psi.hupo.org/ms/mzml}precursor')))) + " spectra read.")
        
        # Operations
        first = 0
        for c in args.charge:
            logging.info("\tMaking charge " + str(c) + "...")
            # new_tree = copy.deepcopy(tree)
            new_tree = mzedit(tree, c, first, adjust) #Todo make copy
            # Write output
            logging.info("\tWriting output file...")
            # outpath = Path(os.path.splitext(infile)[0] + "_ch" + str(c) + ".mzML")
            outpath = os.path.join(outdir, os.path.basename(infile)[:-5] + "_ch" + str(c) + ".mzML")
            new_root = new_tree.getroot()
            prettyPrint(new_root)
            new_tree = ET.ElementTree(new_root)
            with open(outpath, 'wb') as f:
                new_tree.write(f, encoding='utf-8')
            first += 1
        logging.info("\tDone.")
    return

if __name__ == '__main__':

    # multiprocessing.freeze_support()
    # parse arguments
    parser = argparse.ArgumentParser(
        description='mzMLedit',
        epilog='''
        Example:
            python mzMLedit.py

        ''')
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "config/ReFrag.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='MS Data file (mzML)')
    parser.add_argument('-c',  '--charge', default='2,3,4', help='Charges, separated by comma (default: %(default)s)',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-a', '--adjust', action='store_true', help="Adjust m/z values")
    parser.add_argument('-w',  '--n_workers', type=int, default=os.cpu_count(), help='Number of threads/n_workers')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()

    # logging debug level. By default, info level
    # log_file = args.infile[:-4] + '_mzMLedit_log.txt'
    # log_file_debug = args.infile[:-4] + '_mzMLedit_log_debug.txt'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
                            # handlers=[logging.FileHandler(log_file_debug),
                            #           logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')
                            # handlers=[logging.FileHandler(log_file),
                            #           logging.StreamHandler()])

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')