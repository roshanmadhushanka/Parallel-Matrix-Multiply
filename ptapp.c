#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

// Global variables
#define DIM 2000
#define MAX 1000
#define MIN 1
#define THREAD_COUNT 10

// Data structures
double matA[DIM][DIM];
double matB[DIM][DIM];
double matC[DIM][DIM];
double flatA[DIM * DIM];
double flatB[DIM * DIM];

// Method signatures
void populateInputMatrix();
void displayMatrix();
void multiplyMatrix();
void *multiplyChunk(void* threadId);
void copyMatrix();
void *copyChunk(void* threadId);
void verifyCopy();
void verifyMultiplication();

pthread_mutex_t mutex;

int main(){
	populateInputMatrix();
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	copyMatrix();
	multiplyMatrix();
	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	printf("%f\n", elapsed);
	verifyMultiplication();
	return 0;
}

void populateInputMatrix(){
	srandom(time(0)+clock()+random());
	for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			matA[i][j] = rand() % MAX + MIN;
			matB[i][j] = rand() % MAX + MIN;
			matC[i][j] = 0;
		}
	}
}

void displayMatrix(){
	if(DIM < 10){
		//Displaying Matrix A
		printf("Matrix A\n");
		printf("---------------------\n");
		for(int i=0; i<DIM; i++){
			for(int j=0; j<DIM; j++){
				printf("%f\t", matA[i][j]);
			}
			printf("\n");
		}

		//Displaying Matrix B
		printf("Matrix B\n");
		printf("---------------------\n");
		for(int i=0; i<DIM; i++){
			for(int j=0; j<DIM; j++){
				printf("%f\t", matB[i][j]);
			}
			printf("\n");
		}

		//Displaying Matrix C
		printf("Matrix C\n");
		printf("---------------------\n");
		for(int i=0; i<DIM; i++){
			for(int j=0; j<DIM; j++){
				printf("%f\t", matC[i][j]);
			}
			printf("\n");
		}
	} else {
		printf("Matrix size is too large to display\n");
	}
}

void multiplyMatrix(){
	pthread_t* thread_handles = malloc(THREAD_COUNT * sizeof(pthread_t));
	for(long i=0; i<THREAD_COUNT; i++){
		pthread_create(&thread_handles[i], NULL, multiplyChunk, (void*) i);
	}

	//Join all the threads
	for(long i=0; i<THREAD_COUNT; i++){
		pthread_join(thread_handles[i], NULL);
	}

	//Deallocate memory
	free(thread_handles);
	
	return;
}

void *multiplyChunk(void* threadId){
	long id = (long) threadId;
	long i_size = DIM / THREAD_COUNT;
	long i_start = id * i_size;
	double tot;

	for(long i=i_start; i<i_start+i_size; i++){
		for(long j=0; j<DIM; j++){
			tot = 0;
			for(long k=0; k<DIM; k++){
				tot += flatA[i*DIM + k] * flatB[j*DIM + k];
			}
			matC[i][j] = tot;
		}
	}
	return NULL;
}

void copyMatrix(){
	pthread_t* thread_handles = malloc(THREAD_COUNT * sizeof(pthread_t));
	for(long i=0; i<THREAD_COUNT; i++){
		pthread_create(&thread_handles[i], NULL, copyChunk, (void*) i);
	}

	//Join all the threads
	for(long i=0; i<THREAD_COUNT; i++){
		pthread_join(thread_handles[i], NULL);
	}

	//Deallocate memory
	free(thread_handles);
}

void* copyChunk(void* threadId){
	long id = (long) threadId;
	long i_size = DIM / THREAD_COUNT;
	long i_start = id * i_size;
	for(long i=i_start; i<i_start+i_size; i++){
		for(long j=0; j<DIM; j++){
			flatA[i*DIM + j] = matA[i][j];
			flatB[i*DIM + j] = matB[j][i];
		}
	}
	return NULL;
}

void verifyCopy(){
	printf("Verify Copied Arrays\n");
	printf("--------------------\n");
	printf("Dimension : %d\n", DIM);

	int i, j, offset;
	for(i=0; i<DIM; i++){
		for(j=0; j<DIM; j++){
			offset = DIM * i + j;
			if(matA[i][j] != flatA[offset]){
				printf("Mismatched MatrixA!\n");
				return;
			}

			if(matB[j][i] != flatB[offset]){
				printf("Mismatched MatrixB\n");
				return;
			}
		}
	}
	printf("Matrices copied successfuly\n");
}

void verifyMultiplication(){
	printf("Verifying multiplication\n");
	int tot;
	for(int i=0; i<DIM; i++){
		for(int j=0; j<DIM; j++){
			tot = 0;
			for(int k=0; k<DIM; k++){
				tot += matA[i][k] * matB[k][j];
			}

			if(tot != matC[i][j]){
				printf("Resultant matrix not correct\n");
				return;
			}
		}
	}
	printf("Resultant matrix is correct\n");
}
