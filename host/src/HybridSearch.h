#ifndef HYRBIDSEARCH_H_INCLUDED
#define HYBRIDSEARCH_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include "arguments.h"
#include "utils.h"
#include "sequences.h"
#include "submat.h"
#include "FPGAsearch.h"

#define CPU_AVX2_INT8_VECTOR_LENGTH 32
#define CPU_AVX2_INT16_VECTOR_LENGTH 16
#define CPU_AVX2_INT32_VECTOR_LENGTH 8
#define CPU_AVX2_UNROLL_COUNT 10

#define SUBMAT_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH 736

// Computes SW search in hybrid mode using AVX2 instructions in host
void hybrid_search_avx2();

// Computes SW search in hybrid mode using SSE instructions in host
void hybrid_search_sse();

#endif
