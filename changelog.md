___
## v0.4.5

### Date 📅 *2025_03*

### Changes in the detail

+ The output files are saved in the output directory. If the output directory is not provided as an input parameter, a default folder (refrag) is created in the same folder as the input files.

+ tqdm CLI arguments injection attack: We have change to 4.66.3
https://github.com/tqdm/tqdm/releases/tag/v4.66.3

+ Create parameter for protein column name
+ Filter ions only if threshold > 0
+ Revert "disable deisotoping"
+ Disable deisotoping
+ Pause modified fragment checks
+ Always consider NM fragments for ppm errors
+ Sort expSpectrum ions by MZ
+ First ion in MGF format scan
+ Begin ion series on 1
+ In case of tie prefer allowed DM sites
+ Hyperscore groups fragments by charge
+ Do not report a site for NM peptides
+ Do not treat 0 as special case in dm_set
+ Assign ions for NM peptides (tie)
+ Follow the same method as with modified peptides
+ Do not remove disallowed ions
+ Simple deisotoping
+ Fix typo in fragment names
+ Solve score tie using matched intensity
+ Report intensity of matched fragments
+ Fix mgf mode
+ Access scans by number in mzml files

___
## v0.4.4

### Date 📅 *2024_10*

### Changes in the detail

+ Add provisional MSF adaptor that obtain the 'delta_peptide'