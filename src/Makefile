## Diego Mazzieri 0000792583

## Questo Makefile fornisce i seguenti target:
##
## make            compila tutti i sorgenti disponibili
## make clean      cancella i file temporanei e gli eseguibili
## make serial     compila la versione seriale
## make openmp     compila la versione OpenMP
## make cuda       compila la versione CUDA
## make benchmark  compila tutti i sorgenti in versione benchmark

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
EXE_SERIAL:=earthquake
EXE:=$(EXE_OMP) $(EXE_SERIAL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600
OMPFLAGS:=-fopenmp
OMPLIBS:=-lgomp
BENCHFLAGS:=-Wno-unused-but-set-variable
BENCHMACRO:=-DBENCHMARK
NVCC:=nvcc
NVCFLAGS+=-Wno-deprecated-gpu-targets

.PHONY: clean benchmark

ALL: $(EXE)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

serial: $(EXE_SERIAL)

$(EXE_OMP): CFLAGS+=$(OMPFLAGS)
$(EXE_OMP): LDLIBS+=$(OMPLIBS)
openmp: $(EXE_OMP)

cuda: $(EXE_CUDA)

benchmark:
	cc $(CFLAGS) $(BENCHFLAGS) $(BENCHMACRO) $(EXE_SERIAL).c -o $(EXE_SERIAL)
	cc $(CFLAGS) $(OMPFLAGS) $(BENCHFLAGS) $(BENCHMACRO) $(OMPLIBS) $(EXE_OMP).c -o $(EXE_OMP)
	$(NVCC) $(NVCFLAGS) $(BENCHMACRO) $(EXE_CUDA).cu -o $(EXE_CUDA)

clean:
	\rm -f $(EXE) *.o *~ *.png out
