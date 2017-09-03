#include "utils.h"

void merge_scores(int * scores, char ** titles, unsigned long int size) {
	unsigned long int i1 = 0;
	unsigned long int i2 = size / 2;
	unsigned long int it = 0;
	// allocate memory for temporary buffers
	char ** tmp2 = (char **) malloc(size*sizeof(char *));
	int * tmp3 = (int *) malloc (size*sizeof(int));

	while(i1 < size/2 && i2 < size) {
		if (scores[i1] > scores[i2]) {
			tmp2[it] = titles[i1];
			tmp3[it] = scores[i1];
			i1++;
		}
		else {
			tmp2[it] = titles[i2];
			tmp3[it] = scores[i2];
			i2 ++;
		}
		it ++;
	}

	while (i1 < size/2) {
		tmp2[it] = titles[i1];
		tmp3[it] = scores[i1];
	    i1++;
	    it++;
	}
	while (i2 < size) {
		tmp2[it] = titles[i2];
		tmp3[it] = scores[i2];
	    i2++;
	    it++;
	}

	memcpy(titles, tmp2, size*sizeof(char *));
	memcpy(scores, tmp3, size*sizeof(int));

	free(tmp2);
	free(tmp3);

}


void mergesort_scores_serial(int * scores, char ** titles, unsigned long int size) {
	int tmp_score;
	char * tmp_seq;

	if (size == 2) { 
		if (scores[0] <= scores[1]) {
			// swap scores
			tmp_score = scores[0];
			scores[0] = scores[1];
			scores[1] = tmp_score;
			// swap titles
			tmp_seq = titles[0];
			titles[0] = titles[1];
			titles[1] = tmp_seq;
		}
	} else {
		if (size > 2){
			mergesort_scores_serial(scores, titles, size/2);
			mergesort_scores_serial(scores + size/2, titles + size/2, size - size/2);
			merge_scores(scores, titles, size);
		}
	}
}

void sort_scores (int * scores, char ** titles, unsigned long int size, int threads) {
    if ( threads == 1) {
	      mergesort_scores_serial(scores, titles, size);
    }
    else if (threads > 1) {
        #pragma omp parallel sections num_threads(threads)
        {
            #pragma omp section
            sort_scores(scores, titles, size/2, threads/2);
            #pragma omp section
            sort_scores(scores + size/2, titles  + size/2, size-size/2, threads-threads/2);
        }

        merge_scores(scores, titles, size);
    } // threads > 1
}

// Wall time
double dwalltime()
{
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Altera");
  if(platform == NULL) {
    printf("ERROR: Unable to find Altera OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
 /* {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }*/

  // Query the available OpenCL devices.
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &max_num_devices));

  // Create the context.
  context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &status);
  checkError(status, "Failed to create context");

  queues.reset(num_devices);
  kernels.reset(num_devices);
  kernel_events.reset(num_devices*100);

  // Create the command queue.
  for (int i = 0; i<num_devices ; i++){
	  queues[i] = clCreateCommandQueue(context, devices[i], CL_QUEUE_PROFILING_ENABLE, &status);
	  checkError(status, "Failed to create command queue");
  }

  // Create the program.
  std::string binary_file = getBoardBinaryFile(PRECOMPILED_BINARY, devices[0]);
//  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), devices, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  for (int i = 0; i<num_devices ; i++){
	  // Create the kernel - name passed in here must match kernel name in the
	  // original CL file, that was compiled into an AOCX file using the AOC tool
	  const char * kernel_name = "sw";  // Kernel name, as defined in the CL file
	  kernels[i] = clCreateKernel(program, kernel_name, &status);
	  checkError(status, "Failed to create kernel");
  }

  // Adapt the max chunk size according to device global memory

   cl_ulong a;
   clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &a, NULL);

	// divide by SUBMAT_ROWS due to SPs
	a = (a*0.8) / SUBMAT_ROWS;

	max_chunk_size = (a < max_chunk_size ? a : max_chunk_size);



  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernels && kernels[i]) {
      clReleaseKernel(kernels[i]);
    }
    if(queues && queues[i]) {
      clReleaseCommandQueue(queues[i]);
    }
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
void display_device_info( cl_device_id device ) {

   printf("\nQuerying device for info:\n");
   printf("========================\n");
   device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
   device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
   device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
   device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
   device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
   device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
   device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
   device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
   device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
   device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
   device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
   device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
   device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
   device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
   device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
   device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
   device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

/* Error checking */
void checkErr(cl_int err, const char * name)
{
	if (err != CL_SUCCESS) {
		printf("\n ERROR (%d): %s\n",err,name);
		exit(EXIT_FAILURE);
	}
}