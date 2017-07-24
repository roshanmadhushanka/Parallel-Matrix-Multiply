#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>

typedef double TYPE;
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

// Method signatures
TYPE** randomSquareMatrix(int dimension);
TYPE** zeroSquareMatrix(int dimension);
void displaySquareMatrix(TYPE** matrix, int dimension);
void merge(TYPE** matrixA, TYPE** matrixB, int dimension);

// Test cases
double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

// Verify multiplication
void verifyMultiplication(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension);

int main(){
	int dimension = 200;
	TYPE** matrixA = randomSquareMatrix(dimension);
	TYPE** matrixB = randomSquareMatrix(dimension);
	TYPE** matrixC = zeroSquareMatrix(dimension);

	double elapsed = optimizedParallelMultiply(matrixA, matrixB, matrixC, dimension);
	printf("Elapsed %f\n", elapsed);

	return 0;
}

TYPE** randomSquareMatrix(int dimension){
	/*
		Generate 2 dimensional random TYPE matrix.
	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	//Random seed
	srandom(time(0)+clock()+random());
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
		}
	}

	return matrix;
}

TYPE** zeroSquareMatrix(int dimension){
	/*
		Generate 2 dimensional zero TYPE matrix.
	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	//Random seed
	srandom(time(0)+clock()+random());
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

void displaySquareMatrix(TYPE** matrix, int dimension){
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			printf("%f\t", matrix[i][j]);
		}
		printf("\n");
	}
}

double sequentialMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){
	/*
		Sequentiall multiply given input matrices and return resultant matrix
	*/

	struct timeval t0, t1;
	gettimeofday(&t0, 0);

	/* Head */
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			for(int k=0; k<dimension; k++){
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

double parallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){
	/*
		Parallel multiply given input matrices and return resultant matrix
	*/

	struct timeval t0, t1;
	gettimeofday(&t0, 0);

	/* Head */
	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			for(int k=0; k<dimension; k++){
				matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

double optimizedParallelMultiply(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension){
	/*
		Parallel multiply given input matrices using optimal methods and return resultant matrix
	*/

	int i, j, k, iOff, jOff;
	TYPE tot;

	struct timeval t0, t1;
	gettimeofday(&t0, 0);

	/* Head */
	merge(matrixA, matrixB, dimension);
	#pragma omp parallel shared(matrixC) private(i, j, k, iOff, jOff, tot)
	{
		#pragma omp for schedule(static)
		for(i=0; i<dimension; i++){
			iOff = i * dimension;
			for(j=0; j<dimension; j++){
				jOff = j * dimension;
				tot = 0;
				for(k=0; k<dimension; k++){
					tot += flatA[iOff + k] * flatB[jOff + k];
				}
				matrixC[i][j] = tot;
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

void merge(TYPE** matrixA, TYPE** matrixB, int dimension){
	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			flatA[i * dimension + j] = matrixA[i][j];
			flatB[j * dimension + i] = matrixB[i][j];
		}
	}
}

void verifyMultiplication(TYPE** matrixA, TYPE** matrixB, TYPE** result, int dimension){
	/*
		Verify the result of the matrix multiplication
	*/
	printf("Verifying Result\n");
	printf("----------------\n");
	TYPE tot;
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			tot = 0;
			for(int k=0; k<dimension; k++){
				tot += matrixA[i][k] * matrixB[k][j];
			}
			if(tot != result[i][j]){
				printf("Result is incorrect!\n");
				return;
			}
		}
	}
	printf("Result is correct!\n");

}


