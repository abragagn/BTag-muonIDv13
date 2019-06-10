#!/bin/bash
i=0;
max=55;
n=1000000;
while [ "$i" -le "$max" ]; do
  mkdir "s$i";
  cd "s$i";
  skip=$(python -c "print 0+$n*$i");
  echo $'#!/bin/sh' > script.sh
  echo $'#BSUB -o testBd.log' >> script.sh
  echo $'eval `scram runtime -sh`' >> script.sh
  echo "pdTreeAnalyze /lustre/cmswork/abragagn/ntuList/MC2018Lists/BdToJpsiKstar_2018_DCAP.list hist$i.root -v outputFile ntu$i.root -v histoMode RECREATE -v use_gen t -v process BdJPsiKx -n $n -s $skip" >> script.sh
  echo "" >> script.sh
  bsub < script.sh;
  cd ..;
  i=`expr "$i" + 1`;
done
