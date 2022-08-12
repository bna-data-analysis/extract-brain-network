# extract-brain-network
Python package to convert raw functional images to connectivity matrices

SETUP 
1. Install numpy, os, shutil, glob, dcm2niix, nilearn, scipy modules for python programming
2. Install Docker and fmriprep 
	https://docs.docker.com/
	https://fmriprep.org/en/stable/installation.html (Fmriprep version - 20.2.3)

Processing Environment - Windows and/or Ubuntu

MRI dataset:
Access Link for Neurocon and TaoWu Dataset : http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html
Access Link for ABIDE/ADNI/PPMI : https://ida.loni.usc.edu/login.jsp. 

Steps to preprocess neuroimages:
1. Raw fmri data are in DICOM/nifti format (This step is to convert raw MRI data into BIDS format)
	(a) ABIDE_Nifti2BIDS.py - To convert raw MRI data (Nifti format) to BIDS - For ABIDE dataset
	(b) ADNI_PPMI_DCM2BIDS.py - To convert raw MRI data (DICOM format) to BIDS - For PPMI and ADNI dataset
	(c) Neurocon and TaoWu dataset are Nifti files and are BIDS formatted data structure
2. Preprocess BIDS formatted neuroimaging data using fmriprep
	(a) fmriprep_shellscript.sh - Script to execute fmriprep preprocessing
3. Convert preprocessed Nifti images into connectivity matrices
	(a) ConnectivityMatrices.py - Code to generate connectivity matrices
4. Perform experimental analysis 
	(a) nips_paper_experiments.py
	(b) EdgeWeightsStatistics.py

Note: Input/Output location and the required modification are detailed within the python codes



