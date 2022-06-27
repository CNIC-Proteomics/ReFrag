# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 16:55:52 2022

@author: alaguillog
"""

import math
import pandas as pd

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