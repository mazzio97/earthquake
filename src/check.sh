#!/bin/bash

## Diego Mazzieri 0000792583

## Questo script permette di generare immagini analoghe a quella fornita come esempio,
## per tutti i valori di n passati come parametro, per tutte e tre le versioni del programma.
## Lo scopo è quello di verificare, confrontando le tre immagini generate, 
## la correttezza delle soluzioni parallele implementate.

if [ $# -eq 0 ]; then
	echo "Usage: ./check.sh n1 [n2 n3 ...]"
	exit 1
fi

NAME=earthquake
STEPS=100000
TARGET=(serial openmp cuda)
EXE=($NAME omp-$NAME cuda-$NAME)

len=${#TARGET[@]}

make clean # Rimuovo eventuali versioni benchmark del programma
for (( i=0; i<$len; i++ ))
do
	make ${TARGET[$i]}
	for n in $@
	do
		./${EXE[$i]} $STEPS $n > out
		gnuplot plot.gp
		mv $NAME.png $n-${TARGET[$i]}-$NAME.png
	done
done
