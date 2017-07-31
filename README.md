# Parallel-Matrix-Multiply

Description
-----------

Parallel Matrix Multiplication is C language based method that efficiently multiply matrices with the help of parallel-for 
implementation in OpenMP library. Apart from that, performance has further improved by optimizing algorithm to reduce cache 
misses.

Here I have mainly focused on analyzing performance over traditional matrix multiplication approach. 
./pmp <Number of iterations> Number of iterations will determine the sample size that you wnat to verify the result.
The test will run from dimension of 200*200 matrix to 2000*2000 with a step of 200.
