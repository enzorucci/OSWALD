#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <string.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "arguments.h"
#include "utils.h"
#include "sequences.h"
#include "submat.h"
#include "FPGAsearch.h"
#include "HybridSearch.h"

// OpenCL runtime configuration
cl_platform_id platform = NULL;
scoped_array<cl_device_id> devices; // num_devices elements
cl_context context = NULL;
scoped_array<cl_command_queue> queues; // num_devices elements
cl_program program = NULL;
scoped_array<cl_kernel> kernels; // num_devices elements
scoped_array<cl_event> kernel_events;

// Program options
char *sequences_filename=NULL, * queries_filename=NULL, *input_filename=NULL, * output_filename=NULL, *op=NULL;
char * submat=blosum62, submat_name[]="BLOSUM62";
int open_gap=OPEN_GAP, extend_gap=EXTEND_GAP, cpu_threads=CPU_THREADS, execution_mode=HYBRID_MODE, cpu_vector_length=CPU_SSE_INT8_VECTOR_LENGTH, cpu_block_size=CPU_BLOCK_SIZE;
unsigned long int top=TOP, max_chunk_size = MAX_CHUNK_SIZE;
unsigned int num_devices=NUM_DEVICES, max_num_devices=MAX_NUM_DEVICES;
double test_db_percentage = TEST_DB_PERCENTAGE;

// Entry point.
int main(int argc, char * argv[]) {

    /* Process program arguments */
    program_arguments_processing(argc,argv);

    /* Database preprocessing */
    if (strcmp(op,"preprocess") == 0)
		preprocess_db (input_filename,output_filename,cpu_threads);
    else {

		// init device and create kernel
		if(!init()) {
			return -1;
		}

	    if (strcmp(op,"info") == 0) {
			  // Display some device information.
			 for (int i=0; i< max_num_devices ; i++)
				display_device_info(devices[i]);
		} else { 

			if (execution_mode == FPGA_MODE)
				fpga_search ();
			else 
				if (cpu_vector_length == CPU_SSE_INT8_VECTOR_LENGTH)
					hybrid_search_sse ();
				else
					hybrid_search_avx2 ();
		}
	}

	return 0;
}







