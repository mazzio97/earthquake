## Diego Mazzieri 0000792583

## Crea due grafici a partire dal file bench (generabile con lo script bench.sh)
## dove, al variare della dimensione di input n vengono mostrati:
## - i tempi di esecuzione delle tre versioni del programma
## - lo speedup, madiante istogrammi, rispetto alla versione seriale 

set terminal png enhanced notransparent size 800,800
set output "benchmark.png"
set multiplot layout 2, 1 
set title "Execution time (s)" font ",14"
set key top left
plot "bench" using 2:xtic(1) title 'Serial' with linespoints lw 2 lc "red",\
	"bench" using 4:xtic(1) title 'OpenMP' with linespoints lw 2 lc "blue",\
	"bench" using 6:xtic(1) title 'CUDA' with linespoints lw 2 lc "green"
set title "Speedup" font ",14"
set style data histogram
set style histogram clustered
set style fill solid border
set xlabel "Grid Dimension (n)"
set key top left
plot "bench" using 3:xtic(1) title "Serial" lc "red",\
	"bench" using 5:xtic(1) title "OpenMP" lc "blue",\
	"bench" using 7:xtic(1) title "CUDA" lc "green"
unset multiplot
