# Results on Running Time : Double, Row-Major

### Legend

* **CPP_BLOCK 1 1** : plain C++ implementation - baseline

* **NEON 8 1** : NEON with loop unrolling factor 8h4, single thread

* **NEON 8 8** : NEON with loop unrolling factor 8, 8 threads

* **VDSP 1 1** : with **vDSP_mmulD()**, **vDSP_vdivD()**, and **vDSP_vsbmD()**.

* **BLAS 1 1** : the combination of **cblas_dgemv()**, **vDSP_vdivD()**, and **vDSP_vsbmD()**.

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/DOUBLE_MATRIX_ROW_MAJOR_Overview.png"><img src="doc/DOUBLE_MATRIX_ROW_MAJOR_Overview.png" alt="overview" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Overview.png"><img src="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Overview.png" alt="overview" height="600"/></a>

### Remarks

* 'BLAS 1 1' performs best up to the size *(1K, 1K)*, 'NEON 8 8' performs best beyond that size.


## Comparison among NEON Loop unrolling

### Legend

* **NEON 1 1** : NEON with no loop unrolling, single thread

* **NEON 2 1** : NEON with loop unrolling factor 2, single thread

* **NEON 4 1** : NEON with loop unrolling factor 4, single thread

* **NEON 8 1** : NEON with loop unrolling factor 8, single thread

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png"><img src="doc/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png" alt="comparison among neon loop unrolling" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png"><img src="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Loop_Unrolling_relative.png" alt="comparison among neon loop unrolling" height="600"/></a>

### Remarks
There is a clear benefit in using NEON intrinsics and the explicit loop unrolling. The sweet spot seems to be the factor 4.


## Comparison among NEON Multithreads

### Legend

* **NEON 8 1**: NEON with loop unrolling factor 8, single thread

* **NEON 8 2**: NEON with loop unrolling factor 8, 2 threads

* **NEON 8 4**: NEON with loop unrolling factor 8, 4 threads

* **NEON 8 8**: NEON with loop unrolling factor 8, 8 threads

### Plots: Mac Mini M1 2020 8 GB
<a href="doc/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Multithreads_relative.png"><img src="doc/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Multithreads_relative.png" alt="comparison among neon multithreads" height="600"/></a>

### Plots: iPhone 13 mini 256 GB
<a href="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Multithreads_relative.png"><img src="doc_ios/DOUBLE_MATRIX_ROW_MAJOR_Comparison_Among_NEON_Multithreads_relative.png" alt="comparison among neon multithreads" height="600"/></a>

### Remarks

There is a benefit in multithreading the NEON implementation, although it is not significant as in the column-major case.
