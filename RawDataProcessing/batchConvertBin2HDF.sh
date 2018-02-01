#!/bin/bash

CFGFILE=$1
echo "Converting bin2HDF from batch file $1"

XCOORD=`awk 'NR==1 {print $1}' $CFGFILE`
YCOORD=`awk 'NR==1 {print $2}' $CFGFILE`
DATAPATH=`awk 'NR==2' $CFGFILE`
TS=( `awk 'NR==3' $CFGFILE` )
IT=( `awk 'NR==4' $CFGFILE` )
BMPATH=`awk 'NR==5' $CFGFILE`
BMFLAG=`awk 'NR==6' $CFGFILE`
OUTPATH=`awk 'NR==7' $CFGFILE`

TLEN=${#TS[@]}

for ((i=0; i<${TLEN}; i++));
do
  printf "\n-------------------------------\n"
  echo "Starting new file..."
  echo "Timestamp ${TS[$i]}, int time ${IT[$i]}"
  touch tmp.cfg
  printf $XCOORD' '$YCOORD'\n' >> tmp.cfg
  printf $DATAPATH'\n' >> tmp.cfg
  printf ${TS[$i]}'\n' >> tmp.cfg
  printf ${IT[$i]}'\n' >> tmp.cfg
  printf $BMPATH'\n' >> tmp.cfg
  printf $BMFLAG >> tmp.cfg
  printf $OUTPATH >> tmp.cfg

  ./Bin2HDF tmp.cfg
  rm tmp.cfg
  echo "Finished Timestamp ${TS[$i]}, int time ${IT[$i]}"
done
