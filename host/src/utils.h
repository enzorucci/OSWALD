#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "arguments.h"

using namespace aocl_utils;

// OpenCL runtime configuration
#define STRING_BUFFER_LEN 1024
#define PRECOMPILED_BINARY "sw"

bool init();

void cleanup();

void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);

void device_info_uint( cl_device_id device, cl_device_info param, const char* name);

void device_info_bool( cl_device_id device, cl_device_info param, const char* name);

void device_info_string( cl_device_id device, cl_device_info param, const char* name);

void display_device_info( cl_device_id device );

void checkErr(cl_int err, const char * name);

// Scores sorting
void merge_scores(int * scores, char ** titles, unsigned long int size);

void mergesort_scores_serial(int * scores, char ** titles, unsigned long int size);

void sort_scores (int * scores, char ** titles, unsigned long int size, int threads);

double dwalltime();

// OpenCL runtime configuration
extern cl_platform_id platform;
extern scoped_array<cl_device_id> devices; // num_devices elements
extern cl_context context;
extern scoped_array<cl_command_queue> queues; // num_devices elements
extern cl_program program;
extern scoped_array<cl_kernel> kernels; // num_devices elements
extern scoped_array<cl_event> kernel_events;


#endif
