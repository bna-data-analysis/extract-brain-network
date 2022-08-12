# Step by Step procedure to extract connectivity matrices - for a single subject
#Sample TaoWu dataset (INCLUDE SUBJECT NAME) is included in demo folder

STEP 1 - Convert DICOM to BIDS formatted dataset. 
The sample data/subject provided is TaoWu dataset. TaoWu dataset are BIDS formatted dataset. Hence the processing step of DICOM to BIDS conversion is not included for Demo Purpose. 

STEP 2 - Preprocess data using fmriprep.
1. Install fmriprep - https://fmriprep.org/en/stable/installation.html
2. Single line command to run the subject is as follows "fmriprep-docker /Demo/sample_taowu /Demo/sample_taowu/derivatives participant --participant-label control002S0413 --skip-bids-validation --stop-on-first-crash --md-only-boilerplate --fs-no-reconall --output-spaces MNI152NLin2009cAsym:res-2 --fs-license-file /license_location/Freesurfer_license/license.txt --ignore slicetiming"
2.1. /Demo/sample_taowu - Input folder location
  2.2 /Demo/sample_taowu/derivatives - Output folder location
2.3 control002S0413 - Subject details
2.4 /license_location/Freesurfer_license/license.txt - Freesurfer license location (can be obtained while installing fmriprep)

Note - The preprocessed files after fmriprep processing are available in /Demo/sample_taowu/derivatives folder. The files are included for reference.

STEP 3 - Extract Connectivity matrices
1. Run ConnectivityMatrices.py file. This generates the adjacency matrix and features matrix of graph dataset






