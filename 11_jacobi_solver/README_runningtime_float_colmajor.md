# Results on Running Time : Float, Column-Major

## Overview

### Legend

* **CPP_BLOCK 1 1** : plain C++ implementation - baseline

* **NEON 8 1** : NEON with the loop unrolling of factor 8, single thread

* **NEON 8 8** : NEON with the loop unrolling of factor 8, 8 threads

* **BLAS 1 1** : combination of **cblas_sgemv()**, **vDSP_vdiv()**, and **vDSP_vsbm()**.

* **METAL DEFAULT 0 0** : Metal kernel, threads over rows, loop over columns

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/FLOAT_MATRIX_COL_MAJOR_Overview.png"><img src="doc/FLOAT_MATRIX_COL_MAJOR_Overview.png" alt="overview" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/FLOAT_MATRIX_COL_MAJOR_Overview.png"><img src="doc_ios/FLOAT_MATRIX_COL_MAJOR_Overview.png" alt="overview" height="600"/></a>

### Remarks

* 'BLAS 1 1' performs best among the CPU implementations.

* The overhead of METAL implementation is amortized around the size *(2K, 2K)*
 and exceeds the performance of BLAS beyond that size.


## Comparison among NEON Loop unrolling

### Legend

* **NEON 1 1** : NEON with no loop unrolling, single thread

* **NEON 2 1** : NEON with loop unrolling factor 2, single thread

* **NEON 4 1** : NEON with loop unrolling factor 4, single thread

* **NEON 8 1**: NEON with loop unrolling factor 8, single thread

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png"><img src="doc/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png" alt="comparison among neon loop unrolling" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png"><img src="doc_ios/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png" alt="comparison among neon loop unrolling" height="600"/></a>

### Remarks
There is a clear benefit in using NEON intrinsics, and explicit loop unrolling.

## Comparison among NEON Multithreads

### Legend

* **NEON 8 1** : NEON with loop unrolling factor 8, single thread

* **NEON 8 2** : NEON with loop unrolling factor 8, 2 threads

* **NEON 8 4** : NEON with loop unrolling factor 8, 4 threads

* **NEON 8 8** : NEON with loop unrolling factor 8, 8 threads

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Multithreads_relative.png"><img src="doc/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Multithreads_relative.png" alt="comparison among neon multithreads" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Multithreads_relative.png"><img src="doc_ios/FLOAT_MATRIX_COL_MAJOR_Comparison_Among_NEON_Multithreads_relative.png" alt="comparison among neon multithreads" height="600"/></a>

### Remarks

There is a clear benefit in multithreading the NEON implementation.
The overhead of synchronizing the threads is amortized around *(512, 512)*.
