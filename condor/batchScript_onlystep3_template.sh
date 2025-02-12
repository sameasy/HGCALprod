#!/bin/bash

storage=/grid_mnt/data__data.polcms/cms/tarabini/FOLDER
local=/home/llr/cms/tarabini/CMSSW_12_6_0_pre4/src/HGCALprod
folder=$PWD

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SITECONFIG_PATH=/cvmfs/cms.cern.ch/SITECONF/T2_FR_GRIF_LLR/GRIF_LLR

cd $local
eval $(scram ru -sh)
cd $folder

mkdir $storage/STEP3

cp $storage/step2/step2_$1.root .
mv step2_$1.root step2.root

cp $local/run_Step3andAnalyzer.py .
cmsRun run_Step3andAnalyzer.py

mv step3.root step3_$1.root
mv step3_$1.root $storage/STEP3

cd $folder
rm -f *.py *.cc *.txt *.root
