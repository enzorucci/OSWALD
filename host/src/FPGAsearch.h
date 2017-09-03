#ifndef FPGASEARCH_H_INCLUDED
#define FPGASEARCH_H_INCLUDED

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

#define CPU_AVX2_INT8_VECTOR_LENGTH 32
#define CPU_AVX2_INT16_VECTOR_LENGTH 16
#define CPU_AVX2_INT32_VECTOR_LENGTH 8
#define CPU_AVX2_UNROLL_COUNT 10

#define CPU_SSE_INT8_VECTOR_LENGTH 16
#define CPU_SSE_INT16_VECTOR_LENGTH 8
#define CPU_SSE_INT32_VECTOR_LENGTH 4
#define CPU_SSE_UNROLL_COUNT 10

#define SUBMAT_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH 736
#define SUBMAT_ROWS_x_CPU_SSE_INT8_VECTOR_LENGTH 368

#define FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR 4

// compute SW alignment in FPGA
void fpga_search ();

// compute SW alignment in host using SSE instructions
void sw_host (char * a, unsigned short int m, char *b, unsigned short int n, char * submat, int * scores, int * overflow,
				char open_gap, char extend_gap, char * scoreProfile, __m128i * row,  __m128i * maxRow,  __m128i * maxCol, __m128i * lastCol);

#endif
