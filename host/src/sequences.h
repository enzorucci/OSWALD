#ifndef SEQUENCES_H_INCLUDED
#define SEQUENCES_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include "arguments.h"
#include "utils.h"

#define BUFFER_SIZE 1000
#define ALLOCATION_CHUNK 1000
#define AOCL_ALIGNMENT 64
#define DUMMY_ELEMENT 'Z'+1
#define PREPROCESSED_DUMMY_ELEMENT 23



// DB preprocessing
void preprocess_db (char * input_filename, char * out_filename, int n_procs);

// DB loading and chunk assembling for FPGA search
void assemble_multiple_chunks_db (char * sequences_filename, int vector_length, unsigned long int max_buffer_size, unsigned int num_devices, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_count, 
				unsigned long int * vD, char ***ptr_chunk_vect_sequences_db, unsigned int * chunk_count, unsigned int ** ptr_chunk_vect_sequences_db_count, 
				unsigned long int ** ptr_chunk_vect_accum_sequences_db_count, unsigned long int ** ptr_chunk_vD, unsigned long int * max_chunk_vD, 
				unsigned short int *** ptr_chunk_vect_sequences_db_lengths, unsigned short int *** ptr_chunk_nbbs, unsigned int *** ptr_chunk_vect_sequences_db_disp,
				int n_procs);

// DB loading and chunk assembling for tesing in Hybrid search
void assemble_test_chunks (char * sequences_filename, int cpu_vector_length, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, char *** ptr_sequences,
				unsigned short int ** ptr_sequences_lengths, char ** ptr_test_b, unsigned int * test_cpu_sequences_count,
				unsigned short int ** ptr_test_n, unsigned int ** ptr_test_b_disp, char ** ptr_test_chunk_b, unsigned int * test_fpga_sequences_count,
				unsigned short int ** ptr_test_chunk_n, unsigned int ** ptr_test_chunk_b_disp, unsigned long int * test_chunk_vD, unsigned short int ** ptr_test_chunk_nbb,
				double test_db_percentage, unsigned long int max_chunk_size, unsigned int * max_chunk_sequences_count, 
				int n_procs) ;

// DB loading and chunk assembling in hybrid search
void assemble_db_chunks (char ** sequences, unsigned short int * sequences_lengths, unsigned int sequences_count, unsigned long int D,
				char ** ptr_b, unsigned short int ** ptr_n, unsigned long int ** ptr_b_disp, unsigned int * cpu_sequences_count, char *** ptr_chunk_b, 
				unsigned int * chunk_count, unsigned int ** ptr_chunk_sequences_count, unsigned int ** ptr_chunk_accum_sequences_count, 
				unsigned short int *** ptr_chunk_n, unsigned int *** ptr_chunk_b_disp,
				unsigned short int *** ptr_chunk_nbbs, unsigned long int **ptr_chunk_vD, unsigned int * fpga_sequences_count,
				unsigned long int max_chunk_size, unsigned short int Q, double test_db_percentage, unsigned int test_cpu_sequences_count, unsigned int num_devices, 
				unsigned long int test_fpga_D, unsigned long int test_cpu_D,	double test_fpga_time, double test_cpu_time, int cpu_vector_length, int n_procs);

// Load DB headers
void load_database_headers (char * sequences_filename, unsigned long int sequences_count, int max_title_length, char *** ptr_sequences_db_headers);

// load query sequences
void load_query_sequences(char * queries_filename, char ** ptr_query_sequences, char *** ptr_query_headers, unsigned short int **ptr_query_sequences_lengths,
						unsigned long int * query_sequences_count, unsigned long int * ptr_Q, unsigned int ** ptr_query_sequences_disp, int n_procs);

// Functions for parallel sorting
void merge_sequences(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void mergesort_sequences_serial(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void sort_sequences (char ** sequences,  char ** titles, unsigned short int * sequences_lengths, unsigned long int size, int threads);

#endif
