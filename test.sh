#!/bin/bash 

module load hip
clear

rm $1
make $1
sbatch run.sh $1 

sleep 5
echo 
echo "error file: "
cat run.err* 

echo
echo "output file: "
rm output.txt
cat run.out* > output.txt
cat output.txt

rm *.out* *.err*