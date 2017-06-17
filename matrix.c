/* 
http://www.appentra.com/parallel-matrix-matrix-multiplication 
https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef double TYPE;

void parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);
void sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);
TYPE** randomMatrix(int dimension);
TYPE** zeroMatrix(int dimension);
void displayMatrix(TYPE** matrix, int dimension);

int main(int argc, char* argv[]){
	int dimension = strtol(argv[1], NULL, 10);
	TYPE** matrixA = randomMatrix(dimension);
	TYPE** matrixB = randomMatrix(dimension);
	TYPE** resultMatrix = zeroMatrix(dimension);
	parallelMultiply(matrixA, matrixB, resultMatrix, dimension);
	sequentialMultiply(matrixA, matrixB, resultMatrix, dimension);


	// displayMatrix(matrixA, dimension);
	// displayMatrix(matrixB, dimension);
	// displayMatrix(resultMatrix, dimension);
	return 0;
}

void parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	clock_t start, end;
	double cpu_time_used;

	start = clock();

	/* Begin process */
	int i, j, k, n_thread;
	#pragma omp parallel shared(matrixA, matrixB, result) private(i, j, k)
	{ 
		n_thread = omp_get_num_threads();
		#pragma omp for schedule(static)
	    for (i=0; i<dimension; i++)
	    {
	    	int tot;
	        for (j=0; j<dimension; j++)
	        {	
	        	tot = 0;
	            for (k=0; k<dimension; k++){
	                tot += matrixA[i][k] * matrixB[k][j];
	            }
	            result[i][j] = tot;
	        }
	    }
	}
    /* End process */

    end = clock();

	cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC * n_thread);
	printf("Result for Parallel Matrix Multiplication  \n");
	printf("-------------------------------------------\n");
	printf("Dimension  : %d\n", dimension);
	printf("Time taken : %f seconds\n", cpu_time_used);
}

void sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	clock_t start, end;
	double cpu_time_used;

	start = clock();

	/* Begin process */
	int i, j, k, tot;

    for (i=0; i<dimension; i++)
    {
        for (j=0; j<dimension; j++)
        {
        	tot = 0;
            for (k=0; k<dimension; k++){
                tot += matrixA[i][k] * matrixB[k][j] ;
            }
            result[i][j] = tot;
        }
    }
    /* End process */

    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Result for Sequential Matrix Multiplication\n");
	printf("-------------------------------------------\n");
	printf("Dimension  : %d\n", dimension);
	printf("Time taken : %f seconds\n", cpu_time_used);
}

TYPE** randomMatrix(int dimension){
	TYPE** matrix = malloc(dimension * sizeof(TYPE*));
	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	srandom(time(0)+clock()+random());
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = rand() % 100;
		}
	}
	return matrix;
}

TYPE** zeroMatrix(int dimension){
	TYPE** matrix = malloc(dimension * sizeof(TYPE*));
	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	srandom(time(0)+clock()+random());
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = 0;
		}
	}
	return matrix;
}

void displayMatrix(TYPE** matrix, int dimension){
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}


