#!/bin/bash

#PBS -N swimming
#PBS -q workq
#PBS -l nodes=1:ppn=4
##PBS -k oe

#PBS -l walltime=100:59:59
#PBS -l mem=1G
#PBS -V
#PBS -j oe
#PBS -e /home1/weiym/bart/log
#PBS -o /home1/weiym/bart/log
##PBS -J 0-99

set -x
cd $PBS_O_WORKDIR

echo "$PBS_O_WORKDIR"
#echo "$PBS_ARRAY_INDEX"

EXP_ID=`echo $PBS_JOBID | sed 's/\[[^]]*\]//'`
echo $EXP_ID

Rscript bart_fit_main.R > log/STLBart_MDD_13_test.txt


