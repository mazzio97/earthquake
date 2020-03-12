#!/bin/bash

## Diego Mazzieri 0000792583

## Questo script permette di automatizzare la generazione dei grafici
## utili a confrontare le prestazioni delle tre versioni del programma.
## Vista la necessità di effettuare operazioni in virgola mobile, 
## si è fatto largo uso del comando bash bc. 

if [ $# -eq 0 ]; then
	echo "Usage: ./bench.sh n1 [n2 n3 ...]"
	exit 1
fi

NAME=earthquake
TESTS=5
STEPS=2048
EXE=($NAME omp-$NAME cuda-$NAME)

make benchmark
for n in $@
do
	printf "$n "
	TIME_SERIAL=0
	for exec in ${EXE[@]}
	do
		TIME=0
		for i in `seq 1 $TESTS`
		do
			ELAPSED=`./$exec $STEPS $n`
			TIME=`echo $TIME+$ELAPSED | bc`
		done
		TIME=`echo "scale=3; $TIME/$TESTS" | bc` # Average time
		printf "$TIME "
		# Set TIME_SERIAL on the first iteration 
		if [[ $(echo "if ($TIME_SERIAL > 0) 1 else 0" | bc) -eq 0 ]]; then
			TIME_SERIAL=$TIME
		fi
		# Check for no execution time in order to avoid division for 0
		if [[ $(echo "if ($TIME > 0) 1 else 0" | bc) -eq 1 ]]; then
			SPEEDUP=`echo "scale=3; $TIME_SERIAL/$TIME" | bc`
		else
			SPEEDUP=1
		fi
		printf "$SPEEDUP "
	done
	printf "\n"
done > bench
gnuplot benchmark.gp
