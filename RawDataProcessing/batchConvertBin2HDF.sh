#!/bin/bash

CFGFILE=$1
echo "Converting bin2HDF from batch file $1"

DATAPATH=`awk 'NR==1' $CFGFILE`
BMPATH=`awk 'NR==2' $CFGFILE`
BMFLAG=`awk 'NR==3' $CFGFILE`

TS=( `awk 'NR==4' $CFGFILE` )
IT=( `awk 'NR==5' $CFGFILE` )

TLEN=${#TS[@]}

for ((i=0; i<${TLEN}; i++));
do
  echo "\n-------------------------------\n"
  echo "Starting new file..."
  echo "Timestamp ${TS[$i]}, int time ${IT[$i]}"
  touch tmp.cfg
  printf $DATAPATH'\n' >> tmp.cfg
  printf ${TS[$i]}'\n' >> tmp.cfg
  printf ${IT[$i]}'\n' >> tmp.cfg
  printf $BMPATH'\n' >> tmp.cfg
  printf $BMFLAG >> tmp.cfg

  ./Bin2HDF tmp.cfg

  mv $DATAPATH/${TS[$i]}.h5 $DATAPATH/obs_${TS[$i]}.h5
  python ./indexHDF.py $DATAPATH/obs_${TS[$i]}.h5
  rm tmp.cfg
  echo "Finished Timestamp ${TS[$i]}, int time ${IT[$i]}"
done
