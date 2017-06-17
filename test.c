/* 
http://www.appentra.com/parallel-matrix-matrix-multiplication 
https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef double TYPE;

// Main functionalities
double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);
double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);
double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);
TYPE** randomMatrix(int dimension);
TYPE** zeroMatrix(int dimension);
TYPE* zeroFlatMatrix(int dimension);
TYPE* flatMatrix(TYPE** matrix, int dimension);
void displayMatrix(TYPE** matrix, int dimension);
void displayFlatMatrix(TYPE* matrix, int dimension);

// Performance tests
void sequentialMultiplyTest(int iterations);
void parallelMultiplyTest(int iterations);
void optimizedParallelMultiplyTest(int iterations);

int main(int argc, char* argv[]){
	int iterations = strtol(argv[1], NULL, 10);

	TYPE** matrixA = randomMatrix(dimension);
	TYPE** matrixB = randomMatrix(dimension);

	// Loop from 200 * 200 to 2000 * 2000 by step of 200
	for(int dimension=0; dimension<=2000; dimension=dimension+200){
		double* opmLatency = malloc(iterations * sizeof(double));
		for(int i0; i<iterations; i++){
			; 
		}
		TYPE** matrixResult = zeroMatrix(dimension);	
		optimizedParallelMultiply(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		TYPE** matrixResult = zeroFlatMatrix(dimension);
		parallelMultiply(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		TYPE** matrixResult = zeroFlatMatrix(dimension);
		sequentialMultiply(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);
	}

	free(matrixA);
	free(matrixB);

	return 0;
}

/* Main functionalities */

double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	clock_t start, end;
	double cpu_time_used;
	int i, j, k;

	start = clock();

	/* Begin process */

    for (i=0; i<dimension; i++)
    {
        for (j=0; j<dimension; j++)
        {
            for (k=0; k<dimension; k++){
                result[i][j] += matrixA[i][k] * matrixB[k][j] ;
            }
        }
    }
    /* End process */

    end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	return cpu_time_used;
}

double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	clock_t start, end;
	double cpu_time_used;
	int i, j, k, n_thread;

	start = clock();

	/* Begin process */

	#pragma omp parallel shared(matrixA, matrixB, result) private(i, j, k)
	{ 
		n_thread = omp_get_num_threads();
		#pragma omp for schedule(static)
	    for (i=0; i<dimension; i++)
	    {
	        for (j=0; j<dimension; j++)
	        {	
	            for (k=0; k<dimension; k++){
	                result[i][j] += matrixA[i][k] * matrixB[k][j];
	            }
	        }
	    }
	}

    /* End process */

    end = clock();
	cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC * n_thread);

	return cpu_time_used;
}

double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	clock_t start, end;
	double cpu_time_used;
	int i, j, k, tot, n_thread;

	start = clock();

	/* Begining of process */

	TYPE* matrixFlatA = flatMatrix(matrixA, dimension);
	TYPE* matrixFlatB = flatMatrix(matrixB, dimension);

	#pragma omp parallel shared(matrixFlatA, matrixFlatB, result) private(i, j, k, tot)
	{
		n_thread = omp_get_num_threads();
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++){
			for(j=0; j<dimension; j++){
				tot = 0;
				for(k=0; k<dimension; k++){
					tot += matrixFlatA[dimension * i + k] * matrixFlatB[dimension * k + j];
				}
				result[i][j] = tot;
			}
		}
	}

	free(matrixFlatA);
	free(matrixFlatB);

	/* End of process*/

	end = clock();
	cpu_time_used = ((double) (end - start)) / (CLOCKS_PER_SEC * n_thread);

	return cpu_time_used;
}


TYPE** randomMatrix(int dimension){
	TYPE** matrix = malloc(dimension * sizeof(TYPE*));
	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	//Random seed
	srandom(time(0)+clock()+random());

	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = rand() % 1000 + 1;
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

TYPE* zeroFlatMatrix(int dimension){
	TYPE* matrix = malloc(dimension * dimension * sizeof(TYPE));

	srandom(time(0)+clock()+random());
	for(int i=0; i<dimension*dimension; i++){
		matrix[i] = 0;
	}
	return matrix;
}

TYPE* flatMatrix(TYPE** matrix, int dimension){
	TYPE* flatMatrix = malloc(dimension * dimension * sizeof(TYPE));
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			flatMatrix[i * dimension + j] = matrix[i][j];
		}
	}
	return flatMatrix;
}

void displayMatrix(TYPE** matrix, int dimension){
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}

void displayFlatMatrix(TYPE* matrix, int dimension){
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			printf("%f\t", matrix[dimension * i + j]);
		}
		printf("\n");
	}
}

/* Performance tests */
void sequentialMultiplyTest(int iterations){
	TYPE** matrixA = randomMatrix(dimension);
	TYPE** matrixB = randomMatrix(dimension);

	// Loop from 200 * 200 to 2000 * 2000 by step of 200
	for(int dimension=0; dimension<=2000; dimension=dimension+200){
		double* opmLatency = malloc(iterations * sizeof(double));
		for(int i; i<iterations; i++){
			TYPE** matrixResult = zeroFlatMatrix(dimension);
			sequentialMultiply(matrixA, matrixB, matrixResult, dimension);
			free(matrixResult);
		}
	}
	
	free(matrixA);
	free(matrixB);
}

void parallelMultiplyTest(int iterations){

}

void optimizedParallelMultiplyTest(int iterations){

}


