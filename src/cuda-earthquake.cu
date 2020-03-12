/*
 * Diego Mazzieri 0000792583
 * 
 * Versione CUDA del programma earthquake.c
 */

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>     /* rand() */
#include <assert.h>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* dimensione nel caso di thread block 2D */
#define BLKDIM 32
/* dimensione nel caso di thread block 1D */
#define BLKDIM_LINEAR (BLKDIM*BLKDIM)

/**
 * Restituisce un puntatore all'elemento di coordinate (i,j) del
 * dominio grid con n colonne.
 * Utile sia all'host (per fare setup) che al device (per fare increment_energy e propagate_energy).
 */
__device__ __host__ static inline float *IDX( float *grid, int i, int j, int n )
{
    return (grid + i*n + j);
}

/**
 * Restituisce un numero reale pseudocasuale con probabilita' uniforme
 * nell'intervallo [a, b], con a < b.
 */
__host__ float randab( float a, float b )
{
    return a + (b-a)*(rand() / (float)RAND_MAX);
}

/**
 * Inizializza il dominio grid di dimensioni n*n con, nelle celle intermedie, valori di energia
 * scelti con probabilità uniforme nell'intervallo [fmin, fmax], nelle celle ai bordi valori nulli.
 * 
 * |0 0 0 0 0 0 0 0 0 0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0|# # # # # # # #|0|
 * |0 0 0 0 0 0 0 0 0 0|
 * 
 * Non è stato possibile parallelizzare questa funzione con CUDA in quanto rand() non e' thread-safe.
 */
__host__ void setup( float* grid, int ext_n, float fmin, float fmax )
{
    /* La prima riga della matrice è costituita da sole ghost cells */
    for( int j=0; j<ext_n; j++) {
        *IDX(grid, 0, j, ext_n) = 0;
    }
    /* Le successive n-2 righe iniziano e finiscono con una ghost cell 
       mentre hanno valori casuali nelle celle intermedie. */
    for ( int i=1; i<ext_n-1; i++ ) {
        *IDX(grid, i, 0, ext_n) = 0;
        for ( int j=1; j<ext_n-1; j++ ) {
            *IDX(grid, i, j, ext_n) = randab(fmin, fmax);
        }
        *IDX(grid, i, ext_n-1, ext_n) = 0;
    }
    /* Anche l'ultima riga, come la prima, è costituita da sole ghost cells */
    for( int j=0; j<ext_n; j++) {
        *IDX(grid, ext_n-1, j, ext_n) = 0;
    }
}

/**
 * Somma delta a tutte le celle tranne quelle ai bordi del dominio grid di dimensioni
 * n*n. Questo kernel realizza il passo 1 descritto nella specifica del progetto.
 */
 __global__ void increment_energy( float *grid, int ext_n, float delta )
{
    /* Partendo da ext_n salto automaticamente la prima riga composta da sole ghost cells */
    const int i = ext_n + threadIdx.x + blockIdx.x * blockDim.x;
    const int col = i % ext_n;
    
    /* Escludo le ghost cells o le celle al di fuori del dominio dall'incremento */
    if (col > 0 && /* non appartenente alla prima colonna */ 
        col < ext_n-1 && /* non appartenente all'ultima colonna */
        i < ext_n*(ext_n-1)) { /* prima dell'ultima riga */
        grid[i] += delta;
    }
}

/**
 * Calcola il numero di celle la cui energia è strettamente
 * maggiore di EMAX e inserisce il risultato in count.
 */
__global__ void count_cells( float *grid, int ext_size, int *count )
{
    __shared__ int local_sum[BLKDIM_LINEAR];
    int lindex = threadIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;
    /* 1 -> la cella ha energia maggiore della massima 
       0 -> la cella ha energia minore della massima o l'accesso è out of bound */
    local_sum[lindex] = (gindex < ext_size && grid[gindex] > EMAX);
    __syncthreads();
    /* Riduzione dell'array local_sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            local_sum[lindex] += local_sum[lindex + bsize];
        }
        bsize /= 2;
        __syncthreads();
    }
    if ( lindex == 0 ) {
        /* Aggiungo atomicamente a count il risultato della riduzione calcolata nel blocco corrente */
        atomicAdd(count, local_sum[0]);
    }
}

/**
 * Calcola l'energia totale della grid sommando 
 * il valore delle singole celle e inserisce il risultato in sum. 
 */
__global__ void sum_energy( float *grid, int ext_size, float *sum )
{
    __shared__ float local_sum[BLKDIM_LINEAR];
    int lindex = threadIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;
    /* 0 -> accesso out of bound */
    local_sum[lindex] = (gindex < ext_size) ? grid[gindex] : 0;
    __syncthreads();
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            local_sum[lindex] += local_sum[lindex + bsize];
        }
        bsize /= 2;
        __syncthreads();
    }
    if ( lindex == 0 ) {
        atomicAdd(sum, local_sum[0]);
    }
}

/**
 * Sostituisce le energie cinetiche totali calcolate per ogni step
 * con le rispettive energie cinetiche medie.
 */
__global__ void average_energy( float *sums, int nsteps, int size ) 
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if ( i < nsteps ) {
        sums[i] /= size;
    }
}

/** 
 * Distribuisce l'energia di ogni cella a quelle adiacenti.
 * cur denota il dominio corrente, next denota il dominio
 * che conterra' il nuovo valore delle energie. Questa funzione
 * realizza il passo 2 descritto nella specifica del progetto.
 */
__global__ void propagate_energy( float *cur, float *next, int ext_n )
{
    /* Prevedendo una ghost area anche per la memoria shared 
       vengono aggiornate (BLKDIM-2)*(BLKDIM-2) celle 
       ma è possibile semplificare la gestione delle celle ai bordi 
       del dominio gestito da ogni blocco. */
    __shared__ float buf[BLKDIM][BLKDIM];
    
    /* Gli indici globali vanno quindi calcolati come se i blocchi
       fossero di dimensione (BLKDIM-2)*(BLKDIM-2) */  
    const int gi = threadIdx.y + blockIdx.y * (blockDim.y-2);
    const int gj = threadIdx.x + blockIdx.x * (blockDim.x-2);

    const int li = threadIdx.y;
    const int lj = threadIdx.x;
   
    if ( gi < ext_n && gj < ext_n ) {
        buf[li][lj] = *IDX(cur, gi, gj, ext_n);
        float *out = IDX(next, gi, gj, ext_n);
        __syncthreads();
        const float FDELTA = EMAX/4;
        /* Escludere le ghost cells del buf con il controllo su li e lj
           basterebbe nel caso in cui n fosse multiplo di BLKSIZE-2;
           ma per funzionare nel caso generale, devo anche controllare
           che non vengano considerate celle al di fuori del dominio. */
        if ((li > 0) && (li < blockDim.y-1) &&
            (lj > 0) && (lj < blockDim.x-1) &&
            (gi < ext_n-1) && (gj < ext_n-1)) {
            float F = buf[li][lj];
            if (buf[li  ][lj-1] > EMAX) { F += FDELTA; }
            if (buf[li  ][lj+1] > EMAX) { F += FDELTA; }
            if (buf[li-1][lj  ] > EMAX) { F += FDELTA; }
            if (buf[li+1][lj  ] > EMAX) { F += FDELTA; }
    
            if (F > EMAX) {
                F -= EMAX;
            }
    
            *out = F;
        }
    }
}

int main( int argc, char* argv[] )
{
    float *grid;
    float *d_cur, *d_next, *d_sum, *Emean;
    int s, n = 256, ext_n, nsteps = 2048;
    int *c, *d_c;
    srand(19);

    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [nsteps [n]]\n", argv[0]);
        return EXIT_FAILURE;
    }

    if ( argc > 1 ) {
        nsteps = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        n = atoi(argv[2]);
    }

    /* Parametri in input non validi */
    if ( nsteps <= 0 || n <= 0 ) {
        /* Se non vengono effettuate computazioni 
           il tempo di esecuzione è considerato nullo */
    #ifdef BENCHMARK
	    printf("%.4f\n", 0.0);
    #endif
        return EXIT_FAILURE;
    }

    /* Includo le ghost cells nel dominio */
    ext_n = n+2;
    const size_t grid_mem_size = ext_n*ext_n*sizeof(float);
    const size_t c_mem_size = nsteps*sizeof(int);
    const size_t sum_mem_size = nsteps*sizeof(float);
    
    /* Per evitare comunicazioni ad ogni timestep tra GPU e CPU, 
       memorizzo il numero di celle maggiori di EMAX e l'energia media 
       all'interno di due array allocati nella GPU, che trasferisco nella CPU
       solo una volta che la computazione dello stencil è completata */
    cudaMalloc((void **)&d_c, c_mem_size);
    cudaMalloc((void **)&d_sum, sum_mem_size);
    cudaMalloc((void **)&d_cur, grid_mem_size);
    cudaMalloc((void **)&d_next, grid_mem_size);

    c = (int*)malloc(c_mem_size); assert(c);
    Emean = (float*)malloc(sum_mem_size); assert(Emean);
    grid = (float*)malloc(grid_mem_size); assert(grid);

    /* Inizializzo a 0 gli array allocati nella GPU */
    cudaMemset(d_c, 0, c_mem_size);
    cudaMemset(d_sum, 0, sum_mem_size);

    setup(grid, ext_n, 0, EMAX*0.1);

    /* Copio la grid inizializzata dalla CPU alla GPU */
    cudaMemcpy(d_cur, grid, grid_mem_size, cudaMemcpyHostToDevice);

    dim3 linearBlock(BLKDIM_LINEAR);
    dim3 squaredBlock(BLKDIM, BLKDIM);
    dim3 linearGrid((ext_n*ext_n + BLKDIM_LINEAR - 1) / BLKDIM_LINEAR);
    dim3 squaredGridGhost((ext_n + BLKDIM - 3) / (BLKDIM - 2), (ext_n + BLKDIM - 3) / (BLKDIM - 2));
    
    const double tstart = hpc_gettime();
    for (s=0; s<nsteps; s++) {
        /* Per utilizzare il minor numero possibile di blocchi considero un partizionamento 1D */
        increment_energy<<<linearGrid, linearBlock>>>(d_cur, ext_n, EDELTA);
        /* Dovendo effettuare l'operazione di riduzione su una matrice,
           considero un partizionamento 1D del dominio trattandola come fosse un array */ 
        count_cells<<<linearGrid, linearBlock>>>(d_cur, ext_n*ext_n, d_c + s);
        /* Il numero di blocchi va calcolato tenendo in considerazione che 
           la shared memory fa uso anch'essa di ghost cells, quindi ogni blocco 
           aggiorna (BLKDIM-2)*(BLKDIM-2) celle pur avendo thread block di dimensione BLKDIM*BLKDIM*/
        propagate_energy<<<squaredGridGhost, squaredBlock>>>(d_cur, d_next, ext_n);
        /* Stesso ragionamento utilizzato per count_cells */
        sum_energy<<<linearGrid, linearBlock>>>(d_next, ext_n*ext_n, d_sum + s);

        float *tmp = d_cur;
        d_cur = d_next;
        d_next = tmp;
    }
    /* Prima di trasferire alla CPU i risultati sostituisco 
       all'energia cinetica totale calcolata ad ogni iterazione
       la rispettiva energia cinetica media */
    average_energy<<<(nsteps + BLKDIM_LINEAR - 1) / BLKDIM_LINEAR, linearBlock>>>(d_sum, nsteps, n*n);
    cudaMemcpy(c, d_c, c_mem_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Emean, d_sum, sum_mem_size, cudaMemcpyDeviceToHost);
    /* Non è stato necessario utilizzare cudaDeviceSynchronize in quanto cudaMemcpy, implicitamente, 
       blocca la CPU fino a quando le precedenti chiamate CUDA non sono state completate */
    const double elapsed = hpc_gettime() - tstart;

#ifndef BENCHMARK
    for (s=0; s<nsteps; s++) {
        printf("%d %f\n", c[s], Emean[s]);
    }
    double Mupdates = (((double)n)*n/1.0e6)*nsteps; /* milioni di celle aggiornate per ogni secondo di wall clock time */
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);
#else
    printf("%.4f\n", elapsed);
#endif

    free(grid);
    free(c);
    free(Emean);
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_c);
    cudaFree(d_sum);

    return EXIT_SUCCESS;
}
