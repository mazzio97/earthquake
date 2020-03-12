/*
 * Diego Mazzieri 0000792583
 * 
 * Versione OpenMP del programma earthquake.c
 */

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>     /* rand() */
#include <assert.h>

/* energia massima */
#define EMAX 4.0f
/* energia da aggiungere ad ogni timestep */
#define EDELTA 1e-4
/* energia da trasferire ad ogni vicino */
#define FDELTA (EMAX/4)

/**
 * Restituisce un puntatore all'elemento di coordinate (i,j) del
 * dominio grid con n colonne.
 */
static inline float *IDX(float *grid, int i, int j, int n)
{
    return (grid + i*n + j);
}

/**
 * Restituisce un numero reale pseudocasuale con probabilita' uniforme
 * nell'intervallo [a, b], con a < b.
 */
float randab( float a, float b )
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
 * Non è stato possibile parallelizzare questa funzione con OpenMP in quanto rand() non e' thread-safe.
 */
void setup( float* grid, int ext_n, float fmin, float fmax )
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

int main( int argc, char* argv[] )
{
    float *cur, *next;
    int s, i, j, n = 256, ext_n, nsteps = 2048;
    float Emean = 0;
    int c = 0;

    srand(19); /* Inizializzazione del generatore pseudocasuale */
    
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
        printf("0\n");
    #endif
        return EXIT_FAILURE;
    }

    ext_n = n+2; /* Includo le ghost cells nella dimensione della matrice */
    const size_t grid_mem_size = ext_n*ext_n*sizeof(float);

    /* Allochiamo i domini */
    cur = (float*)malloc(grid_mem_size); assert(cur);
    next = (float*)malloc(grid_mem_size); assert(next);

    /* L'energia iniziale di ciascuna cella e' scelta 
       con probabilita' uniforme nell'intervallo [0, EMAX*0.1] */       
    setup(cur, ext_n, 0, EMAX*0.1);    

    const double tstart = hpc_gettime();

#pragma omp parallel default(none) shared(cur, next, ext_n, n, nsteps, c, Emean) private(s, i, j)
{
    for (s=0; s<nsteps; s++) {
        /* Sommo EDELTA a tutte le celle non ai bordi 
           della griglia (passo 1 descritto nella specifica del progetto) 
           e nel frattempo opero la riduzione per controllare 
           quante hanno superato il valore soglia. */
    #pragma omp for collapse(2) reduction(+:c)
        for (i=1; i<ext_n-1; i++) {
            for (j=1; j<ext_n-1; j++) {
                float *F = IDX(cur, i, j, ext_n);
                *F += EDELTA;
                if ( *F > EMAX ) { c++; }
            }
        }
        /* Distribuisco l'energia di ogni cella a quelle adiacenti. 
           cur denota il dominio corrente, 
           next denota il dominio che conterra' il nuovo valore delle energie. 
           Passo 2 descritto nella specifica del progetto. */
    #pragma omp for collapse(2)
        for (i=1; i<ext_n-1; i++) {
            for (j=1; j<ext_n-1; j++) {
                /* L'aggiornamento di una cella, pur dipendendo dai valori di quelle ad essa adiacenti,
                   non modifica nessun valore nella matrice cur e solo il proprio nella matrice next,
                   di conseguenza è completamente indipendente dall'aggiornamento delle altre. */
                float F = *IDX(cur, i, j, ext_n);
                float *out = IDX(next, i, j, ext_n);

                /* Se l'energia del vicino di sinistra e'
                   maggiore di EMAX, allora la cella (i,j) ricevera'
                   energia addizionale FDELTA = EMAX/4.
                   Data la presenza delle ghost cells 
                   non è necessario fare ulteriori controlli. */
                if (*IDX(cur, i, j-1, ext_n) > EMAX) { F += FDELTA; }
                /* Idem per il vicino di destra */
                if (*IDX(cur, i, j+1, ext_n) > EMAX) { F += FDELTA; }
                /* Idem per il vicino in alto */
                if (*IDX(cur, i-1, j, ext_n) > EMAX) { F += FDELTA; }
                /* Idem per il vicino in basso */
                if (*IDX(cur, i+1, j, ext_n) > EMAX) { F += FDELTA; }

                if (F > EMAX) {
                    F -= EMAX;
                }

                *out = F;
            }
        }
        /* Calcolo l'energia totale delle celle nella griglia. */
    #pragma omp for collapse(2) reduction(+:Emean)
        for (i=1; i<ext_n-1; i++) {
            for (j=1; j<ext_n-1; j++) {
                Emean += *IDX(next, i, j, ext_n);
            }
        }

        /* Solo il primo dei thread che raggiunge 
           questo punto è incaricato di eseguirlo. */
    #pragma omp single
        {
            /* Trasformo l'energia totale in energia media */
            Emean /= (n*n);

        #ifndef BENCHMARK
            printf("%d %f\n", c, Emean);
        #endif

            /* Reset delle variabili per il prossimo step */
            Emean = 0;
            c = 0;

            float *tmp = cur;
            cur = next;
            next = tmp;
        } 
    }
} /* Fine del blocco parallelo */

    const double elapsed = hpc_gettime() - tstart;
    
    /* Quando è definita la macro BENCHMARK l'unico valore da stampare è il tempo di esecuzione */
#ifndef BENCHMARK
    double Mupdates = (((double)n)*n/1.0e6)*nsteps; /* milioni di celle aggiornate per ogni secondo di wall clock time */
    fprintf(stderr, "%s : %.4f Mupdates in %.4f seconds (%f Mupd/sec)\n", argv[0], Mupdates, elapsed, Mupdates/elapsed);
#else
    printf("%.4f\n", elapsed);
#endif

    /* Libera la memoria */
    free(cur);
    free(next);

    return EXIT_SUCCESS;
}
