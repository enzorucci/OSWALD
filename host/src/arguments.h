#ifndef ARGUMENTS_H_INCLUDED
#define ARGUMENTS_H_INCLUDED

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <argp.h>
#include "submat.h"

#define VERSION "1.0"

#define OPEN_GAP 10
#define EXTEND_GAP 2
#define TOP 10
#define MAX_CHUNK_SIZE 134217728
#define TEST_DB_PERCENTAGE 0.01

#define FPGA_MODE 0
#define HYBRID_MODE 1

#define NUM_DEVICES 1
#define MAX_NUM_DEVICES 16
#define FPGA_VECTOR_LENGTH 16
#define FPGA_BLOCK_WIDTH 28

#define CPU_THREADS 4
#define CPU_BLOCK_SIZE 256

// Arguments parsing
void program_arguments_processing (int argc, char * argv[]);
static int parse_opt (int key, char *arg, struct argp_state *state);

// Global options
extern char * sequences_filename, * queries_filename, *input_filename, * output_filename, *op, * submat, submat_name[];
extern int cpu_threads, open_gap, extend_gap, cpu_vector_length, execution_mode, cpu_block_size;
extern unsigned int max_num_devices, num_devices;
extern unsigned long int top, max_chunk_size;
extern char blosum45[], blosum50[], blosum62[], blosum80[], blosum90[], pam30[], pam70[], pam250[];
extern double test_db_percentage;

#endif
