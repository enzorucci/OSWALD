#include "HybridSearch.h"


void hybrid_search_avx2 () {

	char * a, ** a_headers, *b, **chunk_b, ** tmp_sequence_headers, **sequence_headers, * scoreProfiles[MAX_NUM_DEVICES], ** sequences;
	char * test_b, * test_chunk_b;
	unsigned short int * m, sequences_db_max_length, ** chunk_n, *n, **chunk_nbb, *sequences_lengths;
	unsigned short int * test_n, * test_chunk_n, * test_chunk_nbb;
	unsigned int * a_disp, chunk_count, ** chunk_b_disp, s_start=0, scores_disp, * test_b_disp, * test_chunk_b_disp;
	int max_title_length, open_extend_gap, num_active_devices;
	int * scores, *tmp_scores[MAX_NUM_DEVICES], *ptr_scores;
	unsigned long int query_sequences_count, Q, sequences_count, D, *b_disp, * chunk_vD, test_chunk_vD;
	unsigned long int i, j, k, q, c, s, d, t, disp_1, disp_2, disp_3;
	unsigned int * chunk_sequences_count, test_cpu_sequences_count, test_fpga_sequences_count, vect_sequences_count;
	unsigned int max_chunk_sequences_count, fpga_sequences_count, cpu_sequences_count, * chunk_accum_sequences_count;
    time_t current_time = time(NULL);
	double workTime, tick, test_fpga_tick, test_cpu_tick, test_cpu_time, test_fpga_time, test_fpga_total_time, test_fpga_kernel_time=0, thread_time[2];
	// CL vars
	cl_int status;
	cl_mem cl_a[MAX_NUM_DEVICES], cl_scores[MAX_NUM_DEVICES], cl_scoreProfiles[MAX_NUM_DEVICES], cl_n[MAX_NUM_DEVICES];
	cl_mem cl_nbb[MAX_NUM_DEVICES], cl_b_disp[MAX_NUM_DEVICES];
	 // Configure work set over which the kernel will execute
	size_t wgSize[3] = {1, 1, 1};
	size_t gSize[3] = {1, 1, 1};
	// AVX2 vars
	__m256i ** rows, ** maxCols, ** maxRows, ** lastCols;
	char ** SPs;

	// Print database search information
	printf("\nOSWALD v%s \n\n",VERSION);
	printf("Database file:\t\t\t%s\n",sequences_filename);

	// Load query sequences
	load_query_sequences(queries_filename,&a,&a_headers,&m,&query_sequences_count,&Q,&a_disp,cpu_threads);

	assemble_test_chunks (sequences_filename, CPU_AVX2_INT8_VECTOR_LENGTH, &sequences_count, &D, &sequences_db_max_length, &max_title_length,
			&sequences, &sequences_lengths, &test_b, &test_cpu_sequences_count, &test_n, &test_b_disp, &test_chunk_b, &test_fpga_sequences_count,
			&test_chunk_n, &test_chunk_b_disp, &test_chunk_vD, &test_chunk_nbb, test_db_percentage, max_chunk_size, &max_chunk_sequences_count, cpu_threads);

	// allocate memory for 32-bit computing
	posix_memalign((void**)&rows, 32, cpu_threads*sizeof(__m256i *));
	posix_memalign((void**)&maxCols, 32, cpu_threads*sizeof(__m256i *));
	posix_memalign((void**)&maxRows, 32, cpu_threads*sizeof(__m256i *));
	posix_memalign((void**)&lastCols, 32, cpu_threads*sizeof(__m256i *));
	posix_memalign((void**)&SPs, 32, cpu_threads*sizeof(char *));
	for (i=0; i<cpu_threads ; i++){
		posix_memalign((void**)&rows[i], 32, (cpu_block_size+1)*sizeof(__m256i));
		posix_memalign((void**)&maxCols[i], 32, (cpu_block_size+1)*sizeof(__m256i));
		posix_memalign((void**)&maxRows[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m256i));
		posix_memalign((void**)&lastCols[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m256i));
		posix_memalign((void**)&SPs[i], 32, sequences_db_max_length*SUBMAT_ROWS_x_CPU_AVX2_INT8_VECTOR_LENGTH*sizeof(char));
	}

	// Print database search information
	printf("Database size:\t\t\t%ld sequences (%ld residues) \n",sequences_count,D);
	printf("Longest database sequence: \t%d residues\n",sequences_db_max_length);
	printf("Substitution matrix:\t\t%s\n",submat_name);
	printf("Gap open penalty:\t\t%d\n",open_gap);
	printf("Gap extend penalty:\t\t%d\n",extend_gap);
	printf("Query filename:\t\t\t%s\n",queries_filename);

	// Allocate buffers 
	top = (sequences_count < top ? sequences_count : top);
	for (d=0; d<num_devices ; d++) {
		posix_memalign((void**)&tmp_scores[d], AOCL_ALIGNMENT, (query_sequences_count*max_chunk_sequences_count*FPGA_VECTOR_LENGTH)*sizeof(int));
		posix_memalign((void**)&scoreProfiles[d], AOCL_ALIGNMENT, max_chunk_size*SUBMAT_ROWS*sizeof(char));
	}
	// allocate scores buffer
	vect_sequences_count = ceil((double) sequences_count / (double) CPU_AVX2_INT8_VECTOR_LENGTH);
	posix_memalign((void**)&scores, 32, query_sequences_count*vect_sequences_count*CPU_AVX2_INT8_VECTOR_LENGTH*sizeof(int));

	// Allow nested parallelism
	omp_set_nested(1);

	for (d=0; d<num_devices ; d++){

		// Create buffers in device 
		cl_a[d] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q* sizeof(char), a, &status);
		checkErr(status,"clCreateBuffer cl_a");
		cl_n[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned short int), NULL, &status);
		checkErr(status,"clCreateBuffer cl_n");
		cl_nbb[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned short int), NULL, &status);
		checkErr(status,"clCreateBuffer cl_nbb");
		cl_b_disp[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned int), NULL, &status);
		checkErr(status,"clCreateBuffer cl_b_disp");
		cl_scoreProfiles[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_size*SUBMAT_ROWS, NULL, &status); // se puede optimizar: quizás el requerimiento es menor
		checkErr(status,"clCreateBuffer cl_scoreProfiles");
		cl_scores[d] = clCreateBuffer(context, CL_MEM_READ_WRITE, query_sequences_count*max_chunk_sequences_count*FPGA_VECTOR_LENGTH*sizeof(int), NULL, &status);
		checkErr(status,"clCreateBuffer cl_scores");

		// Set the kernel arguments
		status = clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &cl_a[d]);
		checkError(status, "Failed to set kernels[d] arg 0");

		open_extend_gap = open_gap + extend_gap;
		char oeg =  open_gap + extend_gap;
		status = clSetKernelArg(kernels[d], 3, sizeof(char), &oeg);
		checkError(status, "Failed to set kernels[d] arg 3");

		char eg = extend_gap;
		status = clSetKernelArg(kernels[d], 4, sizeof(char), &eg);
		checkError(status, "Failed to set kernels[d] arg 4");

		status = clSetKernelArg(kernels[d], 5, sizeof(cl_mem), &cl_scoreProfiles[d]);
		checkError(status, "Failed to set kernels[d] arg 5");
		status = clSetKernelArg(kernels[d], 6, sizeof(cl_mem), &cl_scores[d]);
		checkError(status, "Failed to set kernels[d] arg 6");

		status = clSetKernelArg(kernels[d], 8, sizeof(unsigned int), &s_start);
		checkError(status, "Failed to set kernel arg 8");

		status = clSetKernelArg(kernels[d], 10, sizeof(cl_mem), &cl_n[d]);
		checkError(status, "Failed to set kernels[d] arg 10");

		status = clSetKernelArg(kernels[d], 11, sizeof(cl_mem), &cl_nbb[d]);
		checkError(status, "Failed to set kernels[d] arg 11");

		status = clSetKernelArg(kernels[d], 12, sizeof(cl_mem), &cl_b_disp[d]);
		checkError(status, "Failed to set kernels[d] arg 12");

	}

	test_fpga_tick = dwalltime();

		#pragma omp parallel num_threads(2)
		{

			int tid = omp_get_thread_num();
			unsigned int d, c, t, s, q, i, j;

			// FPGA thread
			#pragma omp single nowait 
			{

				d = 0;

				status = clSetKernelArg(kernels[d], 9, sizeof(unsigned int), &test_fpga_sequences_count);
				checkError(status, "Failed to set kernel arg 9");

				for (s=0;s<test_fpga_sequences_count ; s++) {

					// SSE vars
					__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
					__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;

					char * ptr_b = test_chunk_b + test_chunk_b_disp[s]*FPGA_VECTOR_LENGTH;
					char * ptr_scoreProfile = scoreProfiles[d] + (test_chunk_b_disp[s]*FPGA_VECTOR_LENGTH*SUBMAT_ROWS);

					// build score profile
					unsigned int disp_1 = test_chunk_n[s]*FPGA_VECTOR_LENGTH;
					for (i=0; i< test_chunk_n[s] ;i++ ) {
						unsigned int disp_2 = i*FPGA_VECTOR_LENGTH;
						// indexes
						b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
						// indexes >= 16
						aux1 = _mm_sub_epi8(b_values, v16);
						// indexes < 16
						aux2 = _mm_cmpgt_epi8(b_values,v15);
						aux3 = _mm_and_si128(aux2,vneg32);
						aux4 = _mm_add_epi8(b_values,aux3);
						for (j=0; j< SUBMAT_ROWS; j++) {
							unsigned int disp_3 = j*disp_1;
							tmp = (__m128i *) (submat + j*SUBMAT_COLS);
							auxBlosum[0] = _mm_load_si128(tmp);
							auxBlosum[1] = _mm_load_si128(tmp+1);
							aux2  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
							aux3  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
							aux0 = _mm_add_epi8(aux2,  aux3);
							_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_2+disp_3),   aux0);
						}
					}
				}

				// Copy score profiles to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_scoreProfiles[d],CL_FALSE, 0, test_chunk_vD*SUBMAT_ROWS*sizeof(char),scoreProfiles[d],0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_scoreProfiles");

				// Copy lengths to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_n[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned short int),test_chunk_n,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_n");

				// Copy nbbs to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_nbb[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned short int),test_chunk_nbb,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_nbb");

				// Copy displacement to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_b_disp[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned int),test_chunk_b_disp,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_b_disp");

				// Wait  queue to finish.
				clFinish(queues[d]);

				tick = dwalltime();
				
				for (q=0; q<query_sequences_count ; q++) {

					status = clSetKernelArg(kernels[d], 1, sizeof(unsigned short int), &m[q]);
					checkError(status, "Failed to set kernel arg 1");

					status = clSetKernelArg(kernels[d], 2, sizeof(unsigned int), &a_disp[q]);
					checkError(status, "Failed to set kernel arg 2");

					scores_disp = q*test_fpga_sequences_count;
					status = clSetKernelArg(kernels[d], 7, sizeof(unsigned int), &scores_disp);
					checkError(status, "Failed to set kernel arg 7");

					// Launch the kernel
					status = clEnqueueNDRangeKernel(queues[d], kernels[d], 1, NULL, gSize, wgSize, 0, NULL, &kernel_events[q]);
					checkError(status, "Failed to launch kernel");
				}

				// Wait for all kernels to finish.
				clWaitForEvents(query_sequences_count, kernel_events);
				for (q=0; q<query_sequences_count ; q++)
					clReleaseEvent(kernel_events[q]);
			
				test_fpga_kernel_time = (dwalltime() - tick);

				// Copy alignment scores to host array 
				status = clEnqueueReadBuffer(queues[d], cl_scores[d], CL_TRUE, 0, query_sequences_count*test_fpga_sequences_count*FPGA_VECTOR_LENGTH*sizeof(int), tmp_scores[d], 0, NULL, NULL);
				checkErr(status,"clEnqueueReadBuffer: Couldn't read cl_scores buffer");

				test_fpga_total_time = dwalltime()-test_fpga_tick;

				test_fpga_time = (test_fpga_kernel_time/test_fpga_total_time >= 0.98 ? test_fpga_kernel_time : test_fpga_total_time);

			}

			// CPU thread
			#pragma omp single nowait
			{

				test_cpu_tick = dwalltime();

				#pragma omp parallel num_threads(cpu_threads)
				{
					__m256i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores;
					char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;

					__declspec(align(32)) __m256i score, previous, current, auxLastCol, b_values, blosum_lo, blosum_hi;
					__declspec(align(32)) __m256i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
					__declspec(align(32)) __m256i vextend_gap_epi8 = _mm256_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm256_set1_epi8(open_gap+extend_gap);
					__declspec(align(32)) __m256i vextend_gap_epi16 = _mm256_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm256_set1_epi16(open_gap+extend_gap);
					__declspec(align(32)) __m256i vextend_gap_epi32 = _mm256_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm256_set1_epi32(open_gap+extend_gap);
					__declspec(align(32)) __m256i vzero_epi8 = _mm256_set1_epi8(0), vzero_epi16 = _mm256_set1_epi16(0), vzero_epi32 = _mm256_set1_epi32(0);
					__declspec(align(32)) __m256i v15 = _mm256_set1_epi8(15), vneg32 = _mm256_set1_epi8(-32), v16 = _mm256_set1_epi8(16);
					__declspec(align(32)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767);
					__declspec(align(32)) __m128i aux, auxBlosum[2], *tmp;
				
					unsigned  int i, j, ii, jj, k, disp_1, disp_2, disp_3, dim1, dim2, nbb;
					unsigned long int s, q, t; 
					int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;
				
					int tid = omp_get_thread_num();
				
					// allocate memory for auxiliary buffers
					row = rows[tid];
					maxCol = maxCols[tid];
					maxRow = maxRows[tid];
					lastCol = lastCols[tid];
					scoreProfile = SPs[tid];
						
					// calculate alignment score
					#pragma omp for schedule(dynamic) nowait
					for (s=0; s< test_cpu_sequences_count; s++) {

						ptr_b = test_b + test_b_disp[s];

						// build score profile
						disp_1 = test_n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;
						for (i=0; i< test_n[s] ;i++ ) {
							disp_2 = i*CPU_AVX2_INT8_VECTOR_LENGTH;
							// indexes
							b_values =  _mm256_loadu_si256((__m256i *) (ptr_b + disp_2));
							// indexes >= 16
							aux1 = _mm256_sub_epi8(b_values, v16);
							// indexes < 16
							aux2 = _mm256_cmpgt_epi8(b_values,v15);
							aux3 = _mm256_and_si256(aux2,vneg32);
							aux4 = _mm256_add_epi8(b_values,aux3);
							ptr_scoreProfile = scoreProfile + disp_2;
							for (j=0; j< SUBMAT_ROWS; j++) {
								disp_2 = j*disp_1;
								disp_3 = j*SUBMAT_COLS;
								tmp = (__m128i*) (submat + disp_3);
								auxBlosum[0] = _mm_load_si128(tmp);
								auxBlosum[1] = _mm_load_si128(tmp+1);
								blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
								blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
								aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
								aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
								_mm256_store_si256((__m256i *)(ptr_scoreProfile+disp_2),_mm256_or_si256(aux5,aux6));
							}
						}
				
						// caluclate number of blocks
						nbb = ceil( (double) test_n[s] / (double) cpu_block_size);


						for (q=0; q<query_sequences_count; q++){

							ptr_a = a + a_disp[q];
							ptr_scores = (__m256i *) (scores + (q*vect_sequences_count+s)*CPU_AVX2_INT8_VECTOR_LENGTH);

							// init buffers
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
							#pragma unroll(CPU_AVX2_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
								
							// set score to 0
							score = _mm256_set1_epi8(0);
							// calculate a[i] displacement
							disp_1 = test_n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								dim1 = test_n[s]-k*cpu_block_size;
								dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

								// init buffers
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi8(0);
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi8(0);
								auxLastCol = _mm256_set1_epi8(0);

								for( i = 1; i < m[q]+1; i++){
								
									// previous must start in 0
									previous = _mm256_set1_epi8(0);
									// update row[0] with lastCol[i-1]
									row[0] = lastCol[i-1];
									// calculate score profile displacement
									ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1 + (k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH);
									// calculate dim12
									dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {
											//calcuate the diagonal value
											current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
											// calculate current max value
											current = _mm256_max_epi8(current, maxRow[i]);
											current = _mm256_max_epi8(current, maxCol[j]);
											current = _mm256_max_epi8(current, vzero_epi8);
											// update maxRow and maxCol
											maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
											maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
											aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
											maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
											maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
											// update row buffer
											row[j-1] = previous;
											previous = current;
											// update max score
											score = _mm256_max_epi8(score,current);
										}
									}
									#pragma unroll
									for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
										//calcuate the diagonal value
										current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
										// calculate current max value
										current = _mm256_max_epi8(current, maxRow[i]);
										current = _mm256_max_epi8(current, maxCol[j]);
										current = _mm256_max_epi8(current, vzero_epi8);
										// update maxRow and maxCol
										maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
										maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
										aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
										maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
										maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
										// update row buffer
										row[j-1] = previous;
										previous = current;
										// update max score
										score = _mm256_max_epi8(score,current);
									}
									// update lastCol
									lastCol[i-1] = auxLastCol;
									auxLastCol = current;
								}
							}

							// store max value
							aux = _mm256_extracti128_si256 (score,0);
							_mm256_store_si256 (ptr_scores,_mm256_cvtepi8_epi32(aux));
							_mm256_store_si256 (ptr_scores+1,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));
							aux = _mm256_extracti128_si256 (score,1);
							_mm256_store_si256 (ptr_scores+2,_mm256_cvtepi8_epi32(aux));
							_mm256_store_si256 (ptr_scores+3,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));

							// overflow detection
							aux1 = _mm256_cmpeq_epi8(score,v127);
							overflow_flag =  _mm256_testz_si256(aux1,v127); 

							// if overflow
							if (overflow_flag == 0){

								// check overflow in low 16 bits
								aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,0),0),v127);
								bb1_start = _mm256_testz_si256(aux1,v127);
								// check overflow in high 16 bits
								aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,1),0),v127);
								bb1_end = 2 - _mm256_testz_si256(aux1,v127);

								// recalculate using 16-bit signed integer precision
								for (bb1=bb1_start; bb1<bb1_end ; bb1++){

									// init buffers
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
										
									// set score to 0
									score = _mm256_set1_epi8(0);

									disp_2 = bb1*CPU_AVX2_INT16_VECTOR_LENGTH;

									for (k=0; k < nbb; k++){

										// calculate dim
										dim1 = test_n[s]-k*cpu_block_size;
										dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

										// init buffers
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi16(0);
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi16(0);
										auxLastCol = _mm256_set1_epi16(0);

										for( i = 1; i < m[q]+1; i++){
										
											// previous must start in 0
											previous = _mm256_set1_epi16(0);
											// update row[0] with lastCol[i-1]
											row[0] = lastCol[i-1];
											// calculate score profile displacement
											ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_2 + k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH;
											// calculate dim2
											dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

											for (ii=0; ii<dim2 ; ii++) {

												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {												//calcuate the diagonal value
													current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
													// calculate current max value
													current = _mm256_max_epi16(current, maxRow[i]);
													current = _mm256_max_epi16(current, maxCol[j]);
													current = _mm256_max_epi16(current, vzero_epi16);
													// update maxRow and maxCol
													maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
													maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
													aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
													maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
													maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
													// update row buffer
													row[j-1] = previous;
													previous = current;
													// update max score
													score = _mm256_max_epi16(score,current);
												}
											}
											#pragma unroll
											for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
												current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
												// calculate current max value
												current = _mm256_max_epi16(current, maxRow[i]);
												current = _mm256_max_epi16(current, maxCol[j]);
												current = _mm256_max_epi16(current, vzero_epi16);
												// update maxRow and maxCol
												maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
												maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
												aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
												maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
												maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm256_max_epi16(score,current);
											}
											// update lastCol
											lastCol[i-1] = auxLastCol;
											auxLastCol = current;
										}
									}

									// store max value
									aux = _mm256_extracti128_si256 (score,0);
									_mm256_store_si256 (ptr_scores+bb1*2,_mm256_cvtepi16_epi32(aux));
									aux = _mm256_extracti128_si256 (score,1);
									_mm256_store_si256 (ptr_scores+bb1*2+1,_mm256_cvtepi16_epi32(aux));

									// overflow detection
									aux1 = _mm256_cmpeq_epi16(score,v32767);
									overflow_flag =  _mm256_testz_si256(aux1,v32767); 

									// if overflow
									if (overflow_flag == 0){

										// check overflow in low 16 bits
										aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,0),0),v32767);
										bb2_start = _mm256_testz_si256(aux1,v32767);
										// check overflow in high 16 bits
										aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,1),0),v32767);
										bb2_end = 2 - _mm256_testz_si256(aux1,v32767);

										// recalculate using 32-bit signed integer precision
										for (bb2=bb2_start; bb2<bb2_end ; bb2++){

											// init buffers
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi32(0);
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi32(0);
											
											// set score to 0
											score = _mm256_set1_epi32(0);

											disp_3 = disp_2 + bb2*CPU_AVX2_INT32_VECTOR_LENGTH;

											for (k=0; k < nbb; k++){

												// calculate dim
												dim1 = test_n[s]-k*cpu_block_size;
												dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

												// init buffers
												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi32(0);
												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi32(0);
												auxLastCol = _mm256_set1_epi32(0);

												for( i = 1; i < m[q]+1; i++){
												
													// previous must start in 0
													previous = _mm256_set1_epi32(0);
													// update row[0] with lastCol[i-1]
													row[0] = lastCol[i-1];
													// calculate score profile displacement
													ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_3 + k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH;
													// calculate dim2
													dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

													for (ii=0; ii<dim2 ; ii++) {

														#pragma unroll(CPU_AVX2_UNROLL_COUNT)
														for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {		
															//calcuate the diagonal value
															current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
															// calculate current max value
															current = _mm256_max_epi32(current, maxRow[i]);
															current = _mm256_max_epi32(current, maxCol[j]);
															current = _mm256_max_epi32(current, vzero_epi32);
															// update maxRow and maxCol
															maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
															maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
															aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
															maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
															maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
															// update row buffer
															row[j-1] = previous;
															previous = current;
															// update max score
															score = _mm256_max_epi32(score,current);
														}
													}
													#pragma unroll
													for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
														//calcuate the diagonal value
														current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
														// calculate current max value
														current = _mm256_max_epi32(current, maxRow[i]);
														current = _mm256_max_epi32(current, maxCol[j]);
														current = _mm256_max_epi32(current, vzero_epi32);
														// update maxRow and maxCol
														maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
														maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
														aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
														maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
														maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
														// update row buffer
														row[j-1] = previous;
														previous = current;
														// update max score
														score = _mm256_max_epi32(score,current);
													}

													// update lastCol
													lastCol[i-1] = auxLastCol;
													auxLastCol = current;
												}
											}
											// store max value
											_mm256_store_si256 (ptr_scores+bb1*2+bb2,score);
										}
									}
								}
							}
						}
					}
				}

				test_cpu_time = dwalltime() - test_cpu_tick;

			}

		}

		printf("Test DB percentage:\t\t%.4lf% \n",test_db_percentage);
		printf("CPU estimated speed:\t\t%.2lf GCUPS\n",(Q*test_b_disp[test_cpu_sequences_count]) / (test_cpu_time*1000000000));
		printf("FPGA estimated speed:\t\t%.2lf GCUPS\n",num_devices*(Q*test_chunk_vD) / (test_fpga_total_time*1000000000));

		// Free allocated memory
		free(test_b); free(test_n); free(test_b_disp); free(test_chunk_b); free(test_chunk_n); free(test_chunk_b_disp); free(test_chunk_nbb);

		// dividir BD en dos porciones
		assemble_db_chunks(sequences,sequences_lengths,sequences_count,D,&b,&n,&b_disp,&cpu_sequences_count,&chunk_b,&chunk_count,
					&chunk_sequences_count, &chunk_accum_sequences_count, &chunk_n,&chunk_b_disp,&chunk_nbb,&chunk_vD,&fpga_sequences_count,max_chunk_size,
					Q, test_db_percentage, test_cpu_sequences_count, num_devices, test_chunk_vD, test_b_disp[test_cpu_sequences_count], test_fpga_time,
					test_cpu_time, CPU_AVX2_INT8_VECTOR_LENGTH, cpu_threads);

		tick = dwalltime();

		#pragma omp parallel num_threads(2)
		{
			unsigned int i, j, q, s, t, d, c, current_chunk;

			// FPGA thread
			#pragma omp single nowait 
			{

				s_start = 0;
				status = clSetKernelArg(kernels[d], 8, sizeof(unsigned int), &s_start);
				checkError(status, "Failed to set kernel arg 8");

				for (current_chunk=0; current_chunk < chunk_count; current_chunk+=num_devices){

					num_active_devices = (num_devices < chunk_count-current_chunk ? num_devices : chunk_count-current_chunk);

					for (d=0; d<num_active_devices ; d++){

						c = current_chunk + d;

						status = clSetKernelArg(kernels[d], 9, sizeof(unsigned int), &chunk_sequences_count[c]);
						checkError(status, "Failed to set kernel arg 9");

						for ( s=0;s<chunk_sequences_count[c] ; s++) {

							// SSE vars
							__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
							__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;

							char * ptr_b = chunk_b[c] + (chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH);
							char * ptr_scoreProfile = scoreProfiles[d] + (chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH*SUBMAT_ROWS);

							// build score profile
							unsigned int disp_1 = chunk_n[c][s]*FPGA_VECTOR_LENGTH;
							for (i=0; i< chunk_n[c][s] ;i++ ) {
								unsigned int disp_2 = i*FPGA_VECTOR_LENGTH;
								// indexes
								b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
								// indexes >= 16
								aux1 = _mm_sub_epi8(b_values, v16);
								// indexes < 16
								aux2 = _mm_cmpgt_epi8(b_values,v15);
								aux3 = _mm_and_si128(aux2,vneg32);
								aux4 = _mm_add_epi8(b_values,aux3);
								for (j=0; j< SUBMAT_ROWS; j++) {
									unsigned int disp_3 = j*disp_1;
									tmp = (__m128i *) (submat + j*SUBMAT_COLS);
									auxBlosum[0] = _mm_load_si128(tmp);
									auxBlosum[1] = _mm_load_si128(tmp+1);
									aux2  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
									aux3  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
									aux0 = _mm_add_epi8(aux2,  aux3);
									_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_2+disp_3),   aux0);
								}
							}

						}

						// Copy score profiles to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_scoreProfiles[d],CL_FALSE, 0, chunk_vD[c]*SUBMAT_ROWS*sizeof(char),scoreProfiles[d],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_scoreProfiles");

						// Copy lengths to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_n[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned short int),chunk_n[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_n");

						// Copy nbbs to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_nbb[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned short int),chunk_nbb[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_nbb");

						// Copy displacement to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_b_disp[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned int),chunk_b_disp[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_b_disp");
					}

					// Wait for all queues to finish.
					for(d = 0; d < num_active_devices; d++) 
						clFinish(queues[d]);
					  
					for(d = 0; d < num_active_devices; d++) {

						c = current_chunk+d;

						for (int q=0; q<query_sequences_count ; q++) {

							status = clSetKernelArg(kernels[d], 1, sizeof(unsigned short int), &m[q]);
							checkError(status, "Failed to set kernel arg 1");

							status = clSetKernelArg(kernels[d], 2, sizeof(unsigned int), &a_disp[q]);
							checkError(status, "Failed to set kernel arg 2");

							scores_disp = q*chunk_sequences_count[c];
							status = clSetKernelArg(kernels[d], 7, sizeof(unsigned int), &scores_disp);
							checkError(status, "Failed to set kernel arg 7");

							// Launch the kernel
							status = clEnqueueNDRangeKernel(queues[d], kernels[d], 1, NULL, gSize, wgSize, 0, NULL, &kernel_events[d*query_sequences_count+q]);
							checkError(status, "Failed to launch kernel");
						}
					}

					// Wait for all kernels to finish.
					clWaitForEvents(num_active_devices*query_sequences_count, kernel_events);
					for(i = 0; i < num_active_devices*query_sequences_count; i++) 
						clReleaseEvent(kernel_events[i]);

					for(d = 0; d < num_active_devices; d++) {

						c = current_chunk + d;

						// Copy alignment scores to host array 
						status = clEnqueueReadBuffer(queues[d], cl_scores[d], CL_TRUE, 0, query_sequences_count*chunk_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int), tmp_scores[d], 0, NULL, NULL);
						checkErr(status,"clEnqueueReadBuffer: Couldn't read cl_scores buffer");

						// copy tmp_scores fo final scores buffer
						for (q=0; q<query_sequences_count ; q++) 
							memcpy(scores+q*vect_sequences_count*CPU_AVX2_INT8_VECTOR_LENGTH+chunk_accum_sequences_count[c]*FPGA_VECTOR_LENGTH,tmp_scores[d]+(q*chunk_sequences_count[c])*FPGA_VECTOR_LENGTH,chunk_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int));
					}
				}
			}

			// CPU thread
			#pragma omp single nowait
			{
					// CPU thread: generate new parallel region
					#pragma omp parallel num_threads(cpu_threads)
					{

						__m256i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores;
						char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;

						__declspec(align(32)) __m256i score, previous, current, auxLastCol, b_values, blosum_lo, blosum_hi;
						__declspec(align(32)) __m256i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
						__declspec(align(32)) __m256i vextend_gap_epi8 = _mm256_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm256_set1_epi8(open_gap+extend_gap);
						__declspec(align(32)) __m256i vextend_gap_epi16 = _mm256_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm256_set1_epi16(open_gap+extend_gap);
						__declspec(align(32)) __m256i vextend_gap_epi32 = _mm256_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm256_set1_epi32(open_gap+extend_gap);
						__declspec(align(32)) __m256i vzero_epi8 = _mm256_set1_epi8(0), vzero_epi16 = _mm256_set1_epi16(0), vzero_epi32 = _mm256_set1_epi32(0);
						__declspec(align(32)) __m256i v15 = _mm256_set1_epi8(15), vneg32 = _mm256_set1_epi8(-32), v16 = _mm256_set1_epi8(16);
						__declspec(align(32)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767);
						__declspec(align(32)) __m128i aux, auxBlosum[2], *tmp;

						unsigned  int i, j, ii, jj, k, disp_1, disp_2, disp_3, dim1, dim2, nbb;
						unsigned long int s, q, t; 
						int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;

						int tid = omp_get_thread_num();

						// allocate memory for auxiliary buffers
						row = rows[tid];
						maxCol = maxCols[tid];
						maxRow = maxRows[tid];
						lastCol = lastCols[tid];
						scoreProfile = SPs[tid];
							
						// calculate alignment score
						#pragma omp for schedule(dynamic) nowait
						for (s=0; s< cpu_sequences_count; s++) {

							ptr_b = b + b_disp[s];

							// build score profile
							disp_1 = n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;
							for (i=0; i< n[s] ;i++ ) {
								disp_2 = i*CPU_AVX2_INT8_VECTOR_LENGTH;
								// indexes
								b_values =  _mm256_load_si256((__m256i *) (ptr_b + disp_2));
								// indexes >= 16
								aux1 = _mm256_sub_epi8(b_values, v16);
								// indexes < 16
								aux2 = _mm256_cmpgt_epi8(b_values,v15);
								aux3 = _mm256_and_si256(aux2,vneg32);
								aux4 = _mm256_add_epi8(b_values,aux3);
								ptr_scoreProfile = scoreProfile + disp_2;
								for (j=0; j< SUBMAT_ROWS; j++) {
									disp_2 = j*disp_1;
									disp_3 = j*SUBMAT_COLS;
									tmp = (__m128i*) (submat + disp_3);
									auxBlosum[0] = _mm_load_si128(tmp);
									auxBlosum[1] = _mm_load_si128(tmp+1);
									blosum_lo = _mm256_loadu2_m128i(&auxBlosum[0], &auxBlosum[0]);
									blosum_hi = _mm256_loadu2_m128i(&auxBlosum[1], &auxBlosum[1]);
									aux5 = _mm256_shuffle_epi8(blosum_lo,aux4);
									aux6 = _mm256_shuffle_epi8(blosum_hi,aux1);
									_mm256_store_si256((__m256i *)(ptr_scoreProfile+disp_2),_mm256_or_si256(aux5,aux6));
								}
							}


							// caluclate number of blocks
							nbb = ceil( (double) n[s] / (double) cpu_block_size);

							for (q=0; q<query_sequences_count; q++){

								ptr_a = a + a_disp[q];
								ptr_scores = (__m256i *) (scores + (q*vect_sequences_count+s)*CPU_AVX2_INT8_VECTOR_LENGTH + fpga_sequences_count*FPGA_VECTOR_LENGTH);

								// init buffers
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
								#pragma unroll(CPU_AVX2_UNROLL_COUNT)
								for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
								
								// set score to 0
								score = _mm256_set1_epi8(0);
								// calculate a[i] displacement
								disp_1 = n[s]*CPU_AVX2_INT8_VECTOR_LENGTH;

								for (k=0; k < nbb; k++){

									// calculate dim1
									dim1 = n[s]-k*cpu_block_size;
									dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

									// init buffers
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi8(0);
									#pragma unroll(CPU_AVX2_UNROLL_COUNT)
									for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi8(0);
									auxLastCol = _mm256_set1_epi8(0);

									for( i = 1; i < m[q]+1; i++){
								
										// previous must start in 0
										previous = _mm256_set1_epi8(0);
										// update row[0] with lastCol[i-1]
										row[0] = lastCol[i-1];
										// calculate score profile displacement
										ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1 + (k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH);
										// calculate dim2
										dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

										for (ii=0; ii<dim2 ; ii++) {

											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {		
												//calcuate the diagonal value
												current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
												// calculate current max value
												current = _mm256_max_epi8(current, maxRow[i]);
												current = _mm256_max_epi8(current, maxCol[j]);
												current = _mm256_max_epi8(current, vzero_epi8);
												// update maxRow and maxCol
												maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
												maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
												aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
												maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
												maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm256_max_epi8(score,current);
											}
										}
										#pragma unroll
										for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
											//calcuate the diagonal value
											current =  _mm256_adds_epi8(row[j-1], _mm256_load_si256((__m256i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH)));
											// calculate current max value
											current = _mm256_max_epi8(current, maxRow[i]);
											current = _mm256_max_epi8(current, maxCol[j]);
											current = _mm256_max_epi8(current, vzero_epi8);
											// update maxRow and maxCol
											maxRow[i] =  _mm256_subs_epi8(maxRow[i], vextend_gap_epi8);
											maxCol[j] = _mm256_subs_epi8(maxCol[j], vextend_gap_epi8);
											aux0 =  _mm256_subs_epi8(current, vopen_extend_gap_epi8);
											maxRow[i] = _mm256_max_epi8(maxRow[i], aux0);
											maxCol[j] =  _mm256_max_epi8(maxCol[j], aux0);	
											// update row buffer
											row[j-1] = previous;
											previous = current;
											// update max score
											score = _mm256_max_epi8(score,current);
										}
										// update lastCol
										lastCol[i-1] = auxLastCol;
										auxLastCol = current;
									}
								}

								// store max value
								aux = _mm256_extracti128_si256 (score,0);
								_mm256_store_si256 (ptr_scores,_mm256_cvtepi8_epi32(aux));
								_mm256_store_si256 (ptr_scores+1,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));
								aux = _mm256_extracti128_si256 (score,1);
								_mm256_store_si256 (ptr_scores+2,_mm256_cvtepi8_epi32(aux));
								_mm256_store_si256 (ptr_scores+3,_mm256_cvtepi8_epi32(_mm_srli_si128(aux,8)));

								// overflow detection
								aux1 = _mm256_cmpeq_epi8(score,v127);
								overflow_flag =  _mm256_testz_si256(aux1,v127); 

								// if overflow
								if (overflow_flag == 0){

									// check overflow in low 16 bits
									aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,0),0),v127);
									bb1_start = _mm256_testz_si256(aux1,v127);
									// check overflow in high 16 bits
									aux1 = _mm256_cmpeq_epi8(_mm256_inserti128_si256(vzero_epi8,_mm256_extracti128_si256(score,1),0),v127);
									bb1_end = 2 - _mm256_testz_si256(aux1,v127);

									// recalculate using 16-bit signed integer precision
									for (bb1=bb1_start; bb1<bb1_end ; bb1++){

										// init buffers
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi8(0);
										#pragma unroll(CPU_AVX2_UNROLL_COUNT)
										for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi8(0);
										
										// set score to 0
										score = _mm256_set1_epi8(0);

										disp_2 = bb1*CPU_AVX2_INT16_VECTOR_LENGTH;

										for (k=0; k < nbb; k++){

											// calculate dim1
											dim1 = n[s]-k*cpu_block_size;
											dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

											// init buffers
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi16(0);
											#pragma unroll(CPU_AVX2_UNROLL_COUNT)
											for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi16(0);
											auxLastCol = _mm256_set1_epi16(0);

											for( i = 1; i < m[q]+1; i++){
										
												// previous must start in 0
												previous = _mm256_set1_epi16(0);
												// update row[0] with lastCol[i-1]
												row[0] = lastCol[i-1];
												// calculate score profile displacement
												ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_2 + k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH;
												// calculate dim2
												dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

												for (ii=0; ii<dim2 ; ii++) {

													#pragma unroll(CPU_AVX2_UNROLL_COUNT)
													for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {		
														//calcuate the diagonal value
														current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
														// calculate current max value
														current = _mm256_max_epi16(current, maxRow[i]);
														current = _mm256_max_epi16(current, maxCol[j]);
														current = _mm256_max_epi16(current, vzero_epi16);
														// update maxRow and maxCol
														maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
														maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
														aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
														maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
														maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
														// update row buffer
														row[j-1] = previous;
														previous = current;
														// update max score
														score = _mm256_max_epi16(score,current);
													}
												}
												#pragma unroll
												for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
													//calcuate the diagonal value
													current = _mm256_adds_epi16(row[j-1], _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
													// calculate current max value
													current = _mm256_max_epi16(current, maxRow[i]);
													current = _mm256_max_epi16(current, maxCol[j]);
													current = _mm256_max_epi16(current, vzero_epi16);
													// update maxRow and maxCol
													maxRow[i] = _mm256_subs_epi16(maxRow[i], vextend_gap_epi16);
													maxCol[j] = _mm256_subs_epi16(maxCol[j], vextend_gap_epi16);
													aux0 = _mm256_subs_epi16(current, vopen_extend_gap_epi16);
													maxRow[i] = _mm256_max_epi16(maxRow[i], aux0);
													maxCol[j] =  _mm256_max_epi16(maxCol[j], aux0);	
													// update row buffer
													row[j-1] = previous;
													previous = current;
													// update max score
													score = _mm256_max_epi16(score,current);
												}
												// update lastCol
												lastCol[i-1] = auxLastCol;
												auxLastCol = current;

											}

										}

										// store max value
										aux = _mm256_extracti128_si256 (score,0);
										_mm256_store_si256 (ptr_scores+bb1*2,_mm256_cvtepi16_epi32(aux));
										aux = _mm256_extracti128_si256 (score,1);
										_mm256_store_si256 (ptr_scores+bb1*2+1,_mm256_cvtepi16_epi32(aux));

										// overflow detection
										aux1 = _mm256_cmpeq_epi16(score,v32767);
										overflow_flag =  _mm256_testz_si256(aux1,v32767); 

										// if overflow
										if (overflow_flag == 0){

											// check overflow in low 16 bits
											aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,0),0),v32767);
											bb2_start = _mm256_testz_si256(aux1,v32767);
											// check overflow in high 16 bits
											aux1 = _mm256_cmpeq_epi16(_mm256_inserti128_si256(vzero_epi16,_mm256_extracti128_si256(score,1),0),v32767);
											bb2_end = 2 - _mm256_testz_si256(aux1,v32767);

											// recalculate using 32-bit signed integer precision
											for (bb2=bb2_start; bb2<bb2_end ; bb2++){

												// init buffers
												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm256_set1_epi32(0);
												#pragma unroll(CPU_AVX2_UNROLL_COUNT)
												for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm256_set1_epi32(0);
												
												// set score to 0
												score = _mm256_set1_epi32(0);

												disp_3 = disp_2 + bb2*CPU_AVX2_INT32_VECTOR_LENGTH;

												for (k=0; k < nbb; k++){

													// calculate dim1
													dim1 = n[s]-k*cpu_block_size;
													dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

													// init buffers
													#pragma unroll(CPU_AVX2_UNROLL_COUNT)
													for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm256_set1_epi32(0);
													#pragma unroll(CPU_AVX2_UNROLL_COUNT)
													for (i=0; i<dim1+1 ; i++ ) row[i] = _mm256_set1_epi32(0);
													auxLastCol = _mm256_set1_epi32(0);

													for( i = 1; i < m[q]+1; i++){
												
														// previous must start in 0
														previous = _mm256_set1_epi32(0);
														// update row[0] with lastCol[i-1]
														row[0] = lastCol[i-1];
														// calculate score profile displacement
														ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_3 + k*cpu_block_size*CPU_AVX2_INT8_VECTOR_LENGTH;
														// calculate dim2
														dim2 = dim1 / CPU_AVX2_UNROLL_COUNT;

														for (ii=0; ii<dim2 ; ii++) {

															#pragma unroll(CPU_AVX2_UNROLL_COUNT)
															for( j=ii*CPU_AVX2_UNROLL_COUNT+1, jj=0; jj < CPU_AVX2_UNROLL_COUNT; jj++, j++) {	
																//calcuate the diagonal value
																current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
																// calculate current max value
																current = _mm256_max_epi32(current, maxRow[i]);
																current = _mm256_max_epi32(current, maxCol[j]);
																current = _mm256_max_epi32(current, vzero_epi32);
																// update maxRow and maxCol
																maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
																maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
																aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
																maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
																maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
																// update row buffer
																row[j-1] = previous;
																previous = current;
																// update max score
																score = _mm256_max_epi32(score,current);
															}
														}
														#pragma unroll
														for( j = dim2*CPU_AVX2_UNROLL_COUNT+1; j < dim1+1; j++) {
															//calcuate the diagonal value
															current = _mm256_add_epi32(row[j-1], _mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_AVX2_INT8_VECTOR_LENGTH))));
															// calculate current max value
															current = _mm256_max_epi32(current, maxRow[i]);
															current = _mm256_max_epi32(current, maxCol[j]);
															current = _mm256_max_epi32(current, vzero_epi32);
															// update maxRow and maxCol
															maxRow[i] = _mm256_sub_epi32(maxRow[i], vextend_gap_epi32);
															maxCol[j] = _mm256_sub_epi32(maxCol[j], vextend_gap_epi32);
															aux0 = _mm256_sub_epi32(current, vopen_extend_gap_epi32);
															maxRow[i] = _mm256_max_epi32(maxRow[i], aux0);
															maxCol[j] =  _mm256_max_epi32(maxCol[j], aux0);	
															// update row buffer
															row[j-1] = previous;
															previous = current;
															// update max score
															score = _mm256_max_epi32(score,current);
														}
														// update lastCol
														lastCol[i-1] = auxLastCol;
														auxLastCol = current;
													}
												}
												// store max value
												_mm256_store_si256 (ptr_scores+bb1*2+bb2,score);
											}
										}
									}
								}
							}
						}
					}
			}

		}

		// check overflow
		for (q=0; q< query_sequences_count; q++){

			disp_1 = q*vect_sequences_count*CPU_AVX2_INT8_VECTOR_LENGTH;

			for (c=0; c< chunk_count; c++){

				disp_2 = disp_1 + chunk_accum_sequences_count[c]*FPGA_VECTOR_LENGTH;
		
				#pragma omp parallel private(i,j,ptr_scores) num_threads(cpu_threads)
				{
					int tid = omp_get_thread_num();

					#pragma omp for schedule(dynamic)
					for (s=0;s<chunk_sequences_count[c] ; s++) {

						ptr_scores = scores + disp_2 + s*FPGA_VECTOR_LENGTH;
						int overflow_flag=0, overflow[FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR] = {0};					
						
						for (i=0; i < FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR ; i++) {
							int start = i*CPU_SSE_INT32_VECTOR_LENGTH, end=(i+1)*CPU_SSE_INT32_VECTOR_LENGTH;
							for (j=start; j< end; j++)
								if (ptr_scores[j] == CHAR_MAX)
									overflow[i]++;
							overflow_flag += overflow[i];
						}
						if (overflow_flag > 0) 
							sw_host(a+a_disp[q],m[q],chunk_b[c]+chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH,chunk_n[c][s],submat,ptr_scores,overflow,open_gap,extend_gap,SPs[tid],(__m128i *)rows[tid],(__m128i *)maxRows[tid],(__m128i *)maxCols[tid],(__m128i *)lastCols[tid]);
			
					}
				}
			}

		}


		workTime = dwalltime() - tick;

		// Wait for command queue to complete pending events
		for (d=0; d<num_devices ; d++)
			status = clFinish(queues[d]);
		checkError(status, "Failed to finish");

		// Free allocated memory
		free(a);
		free(a_disp);
		if (chunk_count > 0) {
			free(chunk_b);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_n[i]);
			free(chunk_n);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_nbb[i]);
			free(chunk_nbb);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_b_disp[i]);
			free(chunk_b_disp);
			free(chunk_sequences_count);
			free(chunk_accum_sequences_count);
			free(chunk_vD);
		}
		free(b);
		free(n);
		free(b_disp);

		// Load database headers
		load_database_headers (sequences_filename, sequences_count, max_title_length, &sequence_headers);

		// Print top scores
		tmp_sequence_headers = (char**) malloc(sequences_count*sizeof(char *));
		for (i=0; i<query_sequences_count ; i++ ) {
			memcpy(tmp_sequence_headers,sequence_headers,sequences_count*sizeof(char *));
			sort_scores(scores+i*vect_sequences_count*CPU_AVX2_INT8_VECTOR_LENGTH,tmp_sequence_headers,sequences_count,cpu_threads);
			printf("\nQuery no.\t\t\t%d\n",i+1);
			printf("Query description: \t\t%s\n",a_headers[i]+1);
			printf("Query length:\t\t\t%d residues\n",m[i]);
			printf("\nScore\tSequence description\n");
			for (j=0; j<top; j++) 
				printf("%d\t%s",scores[i*vect_sequences_count*CPU_AVX2_INT8_VECTOR_LENGTH+j],tmp_sequence_headers[j]+1);
		}
		printf("\nSearch date:\t\t\t%s",ctime(&current_time));
		printf("Search time:\t\t\t%lf seconds\n",workTime);
		printf("Search speed:\t\t\t%.2lf GCUPS\n",(Q*D) / ((workTime+(test_fpga_time>test_cpu_time?test_fpga_time:test_cpu_time))*1000000000));
		printf("CPU threads:\t\t\t%d\n",cpu_threads);
		printf("CPU vector length:\t\t%d\n",CPU_SSE_INT8_VECTOR_LENGTH);
		printf("CPU block width:\t\t%d\n",cpu_block_size);
		printf("Number of FPGAs:\t\t%u\n",num_devices);
		printf("FPGA vector length:\t\t%d\n",FPGA_VECTOR_LENGTH);
		printf("FPGA block width:\t\t%d\n",FPGA_BLOCK_WIDTH);
		printf("Max. chunk size in FPGA:\t%ld bytes\n",max_chunk_size);

		// Free allocated memory
		free(m);
		free(scores); 	
		for (i=0; i<query_sequences_count ; i++ ) 
			free(a_headers[i]);
		free(a_headers);
		for (i=0; i<sequences_count ; i++ ) 
			free(sequence_headers[i]);
		free(sequence_headers);
		free(tmp_sequence_headers);
		for (d=0; d<num_devices ; d++) {
			free(tmp_scores[d]);
			free(scoreProfiles[d]);
		}
		for (i=0; i<cpu_threads ; i++) {
			free(rows[i]);
			free(maxRows[i]);
			free(maxCols[i]);
			free(lastCols[i]);
			free(SPs[i]);
		}
		free(rows);
		free(maxRows);
		free(maxCols);
		free(lastCols);
		free(SPs);

		// free FPGA resources
		for (d=0; d<num_devices ; d++){
			clReleaseMemObject(cl_a[d]);
			clReleaseMemObject(cl_n[d]);
			clReleaseMemObject(cl_nbb[d]);
			clReleaseMemObject(cl_b_disp[d]);
			clReleaseMemObject(cl_scoreProfiles[d]);
			clReleaseMemObject(cl_scores[d]);
		}

		// Free the resources allocated
		cleanup();


}

void hybrid_search_sse () {

	char * a, ** a_headers, *b, **chunk_b, ** tmp_sequence_headers, **sequence_headers, * scoreProfiles[MAX_NUM_DEVICES], ** sequences;
	char * test_b, * test_chunk_b;
	unsigned short int * m, sequences_db_max_length, ** chunk_n, *n, **chunk_nbb, *sequences_lengths;
	unsigned short int * test_n, * test_chunk_n, * test_chunk_nbb;
	unsigned int * a_disp, chunk_count, ** chunk_b_disp, s_start=0, scores_disp, * test_b_disp, * test_chunk_b_disp;
	int max_title_length, open_extend_gap, num_active_devices;
	int * scores, *tmp_scores[MAX_NUM_DEVICES], *ptr_scores;
	unsigned long int query_sequences_count, Q, sequences_count, D, *b_disp, * chunk_vD, test_chunk_vD;
	unsigned long int i, j, k, q, c, s, d, t, disp_1, disp_2, disp_3;
	unsigned int * chunk_sequences_count, test_cpu_sequences_count, test_fpga_sequences_count, vect_sequences_count;
	unsigned int max_chunk_sequences_count, fpga_sequences_count, cpu_sequences_count, * chunk_accum_sequences_count;
    time_t current_time = time(NULL);
	double workTime, tick, test_fpga_tick, test_cpu_tick, test_cpu_time, test_fpga_time, test_fpga_total_time, test_fpga_kernel_time=0;
	// CL vars
	cl_int status;
	cl_mem cl_a[MAX_NUM_DEVICES], cl_scores[MAX_NUM_DEVICES], cl_scoreProfiles[MAX_NUM_DEVICES], cl_n[MAX_NUM_DEVICES];
	cl_mem cl_nbb[MAX_NUM_DEVICES], cl_b_disp[MAX_NUM_DEVICES];
	 // Configure work set over which the kernel will execute
	size_t wgSize[3] = {1, 1, 1};
	size_t gSize[3] = {1, 1, 1};
	// SSE vars
	__m128i ** rows, ** maxCols, ** maxRows, ** lastCols;
	char ** SPs;

		// Print database search information
		printf("\nOSWALD v1.0 \n\n");
		printf("Database file:\t\t\t%s\n",sequences_filename);

 		// Load query sequences
		load_query_sequences(queries_filename,&a,&a_headers,&m,&query_sequences_count,&Q,&a_disp,cpu_threads);

		assemble_test_chunks (sequences_filename, CPU_SSE_INT8_VECTOR_LENGTH, &sequences_count, &D, &sequences_db_max_length, &max_title_length,
			&sequences, &sequences_lengths, &test_b, &test_cpu_sequences_count, &test_n, &test_b_disp, &test_chunk_b, &test_fpga_sequences_count,
			&test_chunk_n, &test_chunk_b_disp, &test_chunk_vD, &test_chunk_nbb, test_db_percentage, max_chunk_size, &max_chunk_sequences_count, cpu_threads);

		// allocate memory for 32-bit computing
		posix_memalign((void**)&rows, 32, cpu_threads*sizeof(__m128i *));
		posix_memalign((void**)&maxCols, 32, cpu_threads*sizeof(__m128i *));
		posix_memalign((void**)&maxRows, 32, cpu_threads*sizeof(__m128i *));
		posix_memalign((void**)&lastCols, 32, cpu_threads*sizeof(__m128i *));
		posix_memalign((void**)&SPs, 32, cpu_threads*sizeof(char *));
		for (i=0; i<cpu_threads ; i++){
			posix_memalign((void**)&rows[i], 32, (cpu_block_size+1)*sizeof(__m128i));
			posix_memalign((void**)&maxCols[i], 32, (cpu_block_size+1)*sizeof(__m128i));
			posix_memalign((void**)&maxRows[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m128i));
			posix_memalign((void**)&lastCols[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m128i));
			posix_memalign((void**)&SPs[i], 32, sequences_db_max_length*SUBMAT_ROWS_x_CPU_SSE_INT8_VECTOR_LENGTH*sizeof(char));
		}

		// Print database search information
		printf("Database size:\t\t\t%ld sequences (%ld residues) \n",sequences_count,D);
		printf("Longest database sequence: \t%d residues\n",sequences_db_max_length);
		printf("Substitution matrix:\t\t%s\n",submat_name);
		printf("Gap open penalty:\t\t%d\n",open_gap);
		printf("Gap extend penalty:\t\t%d\n",extend_gap);
		printf("Query filename:\t\t\t%s\n",queries_filename);

		// Allocate buffers 
		top = (sequences_count < top ? sequences_count : top);
		for (d=0; d<num_devices ; d++) {
			posix_memalign((void**)&tmp_scores[d], AOCL_ALIGNMENT, (query_sequences_count*max_chunk_sequences_count*FPGA_VECTOR_LENGTH)*sizeof(int));
			posix_memalign((void**)&scoreProfiles[d], AOCL_ALIGNMENT, max_chunk_size*SUBMAT_ROWS*sizeof(char));
		}
		// allocate scores buffer
		vect_sequences_count = ceil((double) sequences_count / (double) CPU_SSE_INT8_VECTOR_LENGTH);
		posix_memalign((void**)&scores, 32, query_sequences_count*vect_sequences_count*CPU_SSE_INT8_VECTOR_LENGTH*sizeof(int));

		// Allow nested parallelism
		omp_set_nested(1);

		for (d=0; d<num_devices ; d++){

			// Create buffers in device 
			cl_a[d] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q* sizeof(char), a, &status);
			checkErr(status,"clCreateBuffer cl_a");
			cl_n[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned short int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_n");
			cl_nbb[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned short int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_nbb");
			cl_b_disp[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_sequences_count*sizeof(unsigned int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_b_disp");
			cl_scoreProfiles[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_size*SUBMAT_ROWS, NULL, &status); // se puede optimizar: quizás el requerimiento es menor
			checkErr(status,"clCreateBuffer cl_scoreProfiles");
			cl_scores[d] = clCreateBuffer(context, CL_MEM_READ_WRITE, query_sequences_count*max_chunk_sequences_count*FPGA_VECTOR_LENGTH*sizeof(int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_scores");

			// Set the kernel arguments
			status = clSetKernelArg(kernels[d], 0, sizeof(cl_mem), &cl_a[d]);
			checkError(status, "Failed to set kernels[d] arg 0");

			open_extend_gap = open_gap + extend_gap;
			char oeg =  open_gap + extend_gap;
			status = clSetKernelArg(kernels[d], 3, sizeof(char), &oeg);
			checkError(status, "Failed to set kernels[d] arg 3");

			char eg = extend_gap;
			status = clSetKernelArg(kernels[d], 4, sizeof(char), &eg);
			checkError(status, "Failed to set kernels[d] arg 4");

			status = clSetKernelArg(kernels[d], 5, sizeof(cl_mem), &cl_scoreProfiles[d]);
			checkError(status, "Failed to set kernels[d] arg 5");

			status = clSetKernelArg(kernels[d], 6, sizeof(cl_mem), &cl_scores[d]);
			checkError(status, "Failed to set kernels[d] arg 6");

			status = clSetKernelArg(kernels[d], 8, sizeof(unsigned int), &s_start);
			checkError(status, "Failed to set kernel arg 8");

			status = clSetKernelArg(kernels[d], 10, sizeof(cl_mem), &cl_n[d]);
			checkError(status, "Failed to set kernels[d] arg 10");

			status = clSetKernelArg(kernels[d], 11, sizeof(cl_mem), &cl_nbb[d]);
			checkError(status, "Failed to set kernels[d] arg 11");

			status = clSetKernelArg(kernels[d], 12, sizeof(cl_mem), &cl_b_disp[d]);
			checkError(status, "Failed to set kernels[d] arg 12");

		}

		test_fpga_tick = dwalltime();

		#pragma omp parallel num_threads(2)
		{

			int tid = omp_get_thread_num();
			unsigned int d, c, t, s, q, i, j;

			// FPGA thread
			#pragma omp single nowait 
			{

				d = 0;

				status = clSetKernelArg(kernels[d], 9, sizeof(unsigned int), &test_fpga_sequences_count);
				checkError(status, "Failed to set kernel arg 9");

				for (s=0;s<test_fpga_sequences_count ; s++) {

					// SSE vars
					__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
					__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;

					char * ptr_b = test_chunk_b + test_chunk_b_disp[s]*FPGA_VECTOR_LENGTH;
					char * ptr_scoreProfile = scoreProfiles[d] + (test_chunk_b_disp[s]*FPGA_VECTOR_LENGTH*SUBMAT_ROWS);

					// build score profile
					unsigned int disp_1 = test_chunk_n[s]*FPGA_VECTOR_LENGTH;
					for (i=0; i< test_chunk_n[s] ;i++ ) {
						unsigned int disp_2 = i*FPGA_VECTOR_LENGTH;
						// indexes
						b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
						// indexes >= 16
						aux1 = _mm_sub_epi8(b_values, v16);
						// indexes < 16
						aux2 = _mm_cmpgt_epi8(b_values,v15);
						aux3 = _mm_and_si128(aux2,vneg32);
						aux4 = _mm_add_epi8(b_values,aux3);
						for (j=0; j< SUBMAT_ROWS; j++) {
							unsigned int disp_3 = j*disp_1;
							tmp = (__m128i *) (submat + j*SUBMAT_COLS);
							auxBlosum[0] = _mm_load_si128(tmp);
							auxBlosum[1] = _mm_load_si128(tmp+1);
							aux2  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
							aux3  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
							aux0 = _mm_add_epi8(aux2,  aux3);
							_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_2+disp_3),   aux0);
						}
					}
				}

				// Copy score profiles to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_scoreProfiles[d],CL_FALSE, 0, test_chunk_vD*SUBMAT_ROWS*sizeof(char),scoreProfiles[d],0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_scoreProfiles");

				// Copy lengths to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_n[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned short int),test_chunk_n,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_n");

				// Copy nbbs to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_nbb[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned short int),test_chunk_nbb,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_nbb");

				// Copy displacement to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_b_disp[d],CL_FALSE, 0, test_fpga_sequences_count*sizeof(unsigned int),test_chunk_b_disp,0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_b_disp");

				// Wait  queue to finish.
				clFinish(queues[d]);

				tick = dwalltime();
				
				for (q=0; q<query_sequences_count ; q++) {

					status = clSetKernelArg(kernels[d], 1, sizeof(unsigned short int), &m[q]);
					checkError(status, "Failed to set kernel arg 1");

					status = clSetKernelArg(kernels[d], 2, sizeof(unsigned int), &a_disp[q]);
					checkError(status, "Failed to set kernel arg 2");

					scores_disp = q*test_fpga_sequences_count;
					status = clSetKernelArg(kernels[d], 7, sizeof(unsigned int), &scores_disp);
					checkError(status, "Failed to set kernel arg 7");

					// Launch the kernel
					status = clEnqueueNDRangeKernel(queues[d], kernels[d], 1, NULL, gSize, wgSize, 0, NULL, &kernel_events[q]);
					checkError(status, "Failed to launch kernel");
				}

				// Wait for all kernels to finish.
				clWaitForEvents(query_sequences_count, kernel_events);
				for (q=0; q<query_sequences_count ; q++)
					clReleaseEvent(kernel_events[q]);
			
				test_fpga_kernel_time = (dwalltime() - tick);

				// Copy alignment scores to host array 
				status = clEnqueueReadBuffer(queues[d], cl_scores[d], CL_TRUE, 0, query_sequences_count*test_fpga_sequences_count*FPGA_VECTOR_LENGTH*sizeof(int), tmp_scores[d], 0, NULL, NULL);
				checkErr(status,"clEnqueueReadBuffer: Couldn't read cl_scores buffer");

				test_fpga_total_time = dwalltime()-test_fpga_tick;

				test_fpga_time = (test_fpga_kernel_time/test_fpga_total_time >= 0.98 ? test_fpga_kernel_time : test_fpga_total_time);
				//test_fpga_time = test_fpga_kernel_time;

			}

			// CPU thread
			#pragma omp single nowait
			{

				test_cpu_tick = dwalltime();

				#pragma omp parallel num_threads(cpu_threads)
				{
					__m128i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores, *tmp;
					char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;

					__declspec(align(32)) __m128i score, previous, current, auxBlosum[2], auxLastCol, b_values;
					__declspec(align(32)) __m128i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
					__declspec(align(32)) __m128i vextend_gap_epi8 = _mm_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm_set1_epi8(open_gap+extend_gap), vzero_epi8 = _mm_set1_epi8(0);
					__declspec(align(32)) __m128i vextend_gap_epi16 = _mm_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm_set1_epi16(open_gap+extend_gap), vzero_epi16 = _mm_set1_epi16(0);
					__declspec(align(32)) __m128i vextend_gap_epi32 = _mm_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm_set1_epi32(open_gap+extend_gap), vzero_epi32 = _mm_set1_epi32(0);
					__declspec(align(32)) __m128i v127 = _mm_set1_epi8(127), v32767 = _mm_set1_epi16(32767);
					__declspec(align(32)) __m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);

					unsigned  int i, j, ii, jj, k, disp_1, disp_2, disp_3, dim1, dim2, nbb;
					unsigned long int s, q; 
					int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;

					int tid = omp_get_thread_num();

					// allocate memory for auxiliary buffers
					row = rows[tid];
					maxCol = maxCols[tid];
					maxRow = maxRows[tid];
					lastCol = lastCols[tid];
					scoreProfile = SPs[tid];
							
					// calculate alignment score
					#pragma omp for schedule(dynamic) nowait
					for (s=0; s< test_cpu_sequences_count; s++) {

						ptr_b = test_b + test_b_disp[s];

						// build score profile
						disp_1 = test_n[s]*CPU_SSE_INT8_VECTOR_LENGTH;
						for (i=0; i< test_n[s] ;i++ ) {
							disp_2 = i*CPU_SSE_INT8_VECTOR_LENGTH;
							// indexes
							b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
							// indexes >= 16
							aux1 = _mm_sub_epi8(b_values, v16);
							// indexes < 16
							aux2 = _mm_cmpgt_epi8(b_values,v15);
							aux3 = _mm_and_si128(aux2,vneg32);
							aux4 = _mm_add_epi8(b_values,aux3);
							ptr_scoreProfile = scoreProfile + disp_2;
							for (j=0; j< SUBMAT_ROWS; j++) {
								disp_3 = j*disp_1;
								tmp = (__m128i *) (submat + j*SUBMAT_COLS);
								auxBlosum[0] = _mm_load_si128(tmp);
								auxBlosum[1] = _mm_load_si128(tmp+1);
								aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
								aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
								aux7 = _mm_add_epi8(aux5,  aux6);
								_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_3),   aux7);
							}
						}

						// caluclate number of blocks
						nbb = ceil( (double) test_n[s] / (double) cpu_block_size);

						for (q=0; q<query_sequences_count; q++){

							ptr_a = a + a_disp[q];
							ptr_scores = (__m128i *) (scores + (q*vect_sequences_count+s)*CPU_SSE_INT8_VECTOR_LENGTH);

							// init buffers
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi8(0);
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi8(0);
								
							// set score to 0
							score = _mm_set1_epi8(0);
							// calculate a[i] displacement
							disp_1 = test_n[s]*CPU_SSE_INT8_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								dim1 = test_n[s]-k*cpu_block_size;
								dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

								// init buffers
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi8(0);
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi8(0);
								auxLastCol = _mm_set1_epi8(0);

								for( i = 1; i < m[q]+1; i++){
								
									// previous must start in 0
									previous = _mm_set1_epi8(0);
									// update row[0] with lastCol[i-1]
									row[0] = lastCol[i-1];
									// calculate score profile displacement
									ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1 + (k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH);
									// calculate dim2
									dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
											//calcuate the diagonal value
											current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
											// calculate current max value
											current = _mm_max_epi8(current, maxRow[i]);
											current = _mm_max_epi8(current, maxCol[j]);
											current = _mm_max_epi8(current, vzero_epi8);
											// update maxRow and maxCol
											maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
											maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
											aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
											maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
											maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
											// update max score
											score = _mm_max_epi8(score,current);
											// update row buffer
											row[j-1] = previous;
											previous = current;
										}
									}
									#pragma unroll
									for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
										//calcuate the diagonal value
										current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
										// calculate current max value
										current = _mm_max_epi8(current, maxRow[i]);
										current = _mm_max_epi8(current, maxCol[j]);
										current = _mm_max_epi8(current, vzero_epi8);
										// update maxRow and maxCol
										maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
										maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
										aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
										maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
										maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
										// update max score
										score = _mm_max_epi8(score,current);
										// update row buffer
										row[j-1] = previous;
										previous = current;
									}

									// update lastCol
									lastCol[i-1] = auxLastCol;
									auxLastCol = current;
								}
							}

							// store max value
							_mm_store_si128 (ptr_scores,_mm_cvtepi8_epi32(score));
							_mm_store_si128 (ptr_scores+1,_mm_cvtepi8_epi32(_mm_srli_si128(score,4)));
							_mm_store_si128 (ptr_scores+2,_mm_cvtepi8_epi32(_mm_srli_si128(score,8)));
							_mm_store_si128 (ptr_scores+3,_mm_cvtepi8_epi32(_mm_srli_si128(score,12)));

							// overflow detection
							aux1 = _mm_cmpeq_epi8(score,v127);
							overflow_flag = _mm_test_all_zeros(aux1,v127); 

							// if overflow
							if (overflow_flag == 0){

								// detect if overflow occurred in low-half, high-half or both halves
								aux1 = _mm_cmpeq_epi8(_mm_slli_si128(score,8),v127);
								bb1_start = _mm_test_all_zeros(aux1,v127);
								aux1 = _mm_cmpeq_epi8(_mm_srli_si128(score,8),v127);
								bb1_end = 2 - _mm_test_all_zeros(aux1,v127);

								// recalculate using 16-bit signed integer precision
								for (bb1=bb1_start; bb1<bb1_end ; bb1++){

									// init buffers
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi16(0);
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi16(0);
									
									// set score to 0
									score = _mm_set1_epi16(0);

									disp_2 = bb1*CPU_SSE_INT16_VECTOR_LENGTH;

									for (k=0; k < nbb; k++){

										// calculate dim1
										dim1 = test_n[s]-k*cpu_block_size;
										dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

										// init buffers
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi16(0);
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi16(0);
										auxLastCol = _mm_set1_epi16(0);

										for( i = 1; i < m[q]+1; i++){
										
											// previous must start in 0
											previous = _mm_set1_epi16(0);
											// update row[0] with lastCol[i-1]
											row[0] = lastCol[i-1];
											// calculate score profile displacement
											ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_2 + k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH;
											// calculate dim2
											dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

											for (ii=0; ii<dim2 ; ii++) {

												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
													//calcuate the diagonal value
													current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
													// calculate current max value
													current = _mm_max_epi16(current, maxRow[i]);
													current = _mm_max_epi16(current, maxCol[j]);
													current = _mm_max_epi16(current, vzero_epi16);
													// update maxRow and maxCol
													maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
													maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
													aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
													maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
													maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
													// update row buffer
													row[j-1] = previous;
													previous = current;
													// update max score
													score = _mm_max_epi16(score,current);
												}
											}
											#pragma unroll
											for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
												//calcuate the diagonal value
												current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
												// calculate current max value
												current = _mm_max_epi16(current, maxRow[i]);
												current = _mm_max_epi16(current, maxCol[j]);
												current = _mm_max_epi16(current, vzero_epi16);
												// update maxRow and maxCol
												maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
												maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
												aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
												maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
												maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm_max_epi16(score,current);
											}
											// update lastCol
											lastCol[i-1] = auxLastCol;
											auxLastCol = current;
										}

									}
									// store max value
									_mm_store_si128 (ptr_scores+bb1*2,_mm_cvtepi16_epi32(score));
									_mm_store_si128 (ptr_scores+bb1*2+1,_mm_cvtepi16_epi32(_mm_srli_si128(score,8)));

									// overflow detection
									aux1 = _mm_cmpeq_epi16(score,v32767);
									overflow_flag = _mm_test_all_zeros(aux1,v32767); 

									// if overflow
									if (overflow_flag == 0){

										// detect if overflow occurred in low-half, high-half or both halves
										aux1 = _mm_cmpeq_epi16(_mm_slli_si128(score,8),v32767);
										bb2_start = _mm_test_all_zeros(aux1,v32767);
										aux1 = _mm_cmpeq_epi16(_mm_srli_si128(score,8),v32767);
										bb2_end = 2 - _mm_test_all_zeros(aux1,v32767);

										// recalculate using 32-bit signed integer precision
										for (bb2=bb2_start; bb2<bb2_end ; bb2++){

											// init buffers
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi32(0);
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi32(0);
											
											// set score to 0
											score = _mm_set1_epi32(0);

											disp_3 = disp_2 + bb2*CPU_SSE_INT32_VECTOR_LENGTH;

											for (k=0; k < nbb; k++){

												// calculate dim1
												dim1 = test_n[s]-k*cpu_block_size;
												dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

												// init buffers
												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi32(0);
												auxLastCol = _mm_set1_epi32(0);

												for( i = 1; i < m[q]+1; i++){
												
													// previous must start in 0
													previous = _mm_set1_epi32(0);
													// update row[0] with lastCol[i-1]
													row[0] = lastCol[i-1];
													// calculate score profile displacement
													ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_3 + k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH;
													// calculate dim2
													dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

													for (ii=0; ii<dim2 ; ii++) {

														#pragma unroll(CPU_SSE_UNROLL_COUNT)
														for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
															//calcuate the diagonal value
															current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
															// calculate current max value
															current = _mm_max_epi32(current, maxRow[i]);
															current = _mm_max_epi32(current, maxCol[j]);
															current = _mm_max_epi32(current, vzero_epi32);
															// update maxRow and maxCol
															maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
															maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
															aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
															maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
															maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
															// update row buffer
															row[j-1] = previous;
															previous = current;
															// update max score
															score = _mm_max_epi32(score,current);
														}
													}
													#pragma unroll
													for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
														//calcuate the diagonal value
														current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
														// calculate current max value
														current = _mm_max_epi32(current, maxRow[i]);
														current = _mm_max_epi32(current, maxCol[j]);
														current = _mm_max_epi32(current, vzero_epi32);
														// update maxRow and maxCol
														maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
														maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
														aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
														maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
														maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
														// update row buffer
														row[j-1] = previous;
														previous = current;
														// update max score
														score = _mm_max_epi32(score,current);
													}
													// update lastCol
													lastCol[i-1] = auxLastCol;
													auxLastCol = current;
												}
											}
											// store max value
											_mm_store_si128 (ptr_scores+bb1*2+bb2,score);
										}
									}
								}
							}
						}
					}
				}

				test_cpu_time = dwalltime() - test_cpu_tick;

			}

		}

		printf("Test DB percentage:\t\t%.4lf% \n",test_db_percentage);
		printf("CPU estimated speed:\t\t%.2lf GCUPS\n",(Q*test_b_disp[test_cpu_sequences_count]) / (test_cpu_time*1000000000));
		printf("FPGA estimated speed:\t\t%.2lf GCUPS\n",num_devices*(Q*test_chunk_vD) / (test_fpga_total_time*1000000000));

		// Free allocated memory
		free(test_b); free(test_n); free(test_b_disp); free(test_chunk_b); free(test_chunk_n); free(test_chunk_b_disp); free(test_chunk_nbb);

		// dividir BD en dos porciones
		assemble_db_chunks(sequences,sequences_lengths,sequences_count,D,&b,&n,&b_disp,&cpu_sequences_count,&chunk_b,&chunk_count,
					&chunk_sequences_count, &chunk_accum_sequences_count, &chunk_n,&chunk_b_disp,&chunk_nbb,&chunk_vD,&fpga_sequences_count,max_chunk_size,
					Q, test_db_percentage, test_cpu_sequences_count, num_devices, test_chunk_vD, test_b_disp[test_cpu_sequences_count], test_fpga_time,
					test_cpu_time, CPU_SSE_INT8_VECTOR_LENGTH, cpu_threads);

		tick = dwalltime();

		#pragma omp parallel num_threads(2)
		{
			unsigned int i, j, q, s, t, d, c, current_chunk;

			// FPGA thread
			#pragma omp single nowait 
			{

				s_start = 0;
				status = clSetKernelArg(kernels[d], 8, sizeof(unsigned int), &s_start);
				checkError(status, "Failed to set kernel arg 8");

				for (current_chunk=0; current_chunk < chunk_count; current_chunk+=num_devices){

					num_active_devices = (num_devices < chunk_count-current_chunk ? num_devices : chunk_count-current_chunk);

					for (d=0; d<num_active_devices ; d++){

						c = current_chunk + d;

						status = clSetKernelArg(kernels[d], 9, sizeof(unsigned int), &chunk_sequences_count[c]);
						checkError(status, "Failed to set kernel arg 9");

						for ( s=0;s<chunk_sequences_count[c] ; s++) {

							// SSE vars
							__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
							__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;

							char * ptr_b = chunk_b[c] + (chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH);
							char * ptr_scoreProfile = scoreProfiles[d] + (chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH*SUBMAT_ROWS);

							// build score profile
							unsigned int disp_1 = chunk_n[c][s]*FPGA_VECTOR_LENGTH;
							for (i=0; i< chunk_n[c][s] ;i++ ) {
								unsigned int disp_2 = i*FPGA_VECTOR_LENGTH;
								// indexes
								b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
								// indexes >= 16
								aux1 = _mm_sub_epi8(b_values, v16);
								// indexes < 16
								aux2 = _mm_cmpgt_epi8(b_values,v15);
								aux3 = _mm_and_si128(aux2,vneg32);
								aux4 = _mm_add_epi8(b_values,aux3);
								for (j=0; j< SUBMAT_ROWS; j++) {
									unsigned int disp_3 = j*disp_1;
									tmp = (__m128i *) (submat + j*SUBMAT_COLS);
									auxBlosum[0] = _mm_load_si128(tmp);
									auxBlosum[1] = _mm_load_si128(tmp+1);
									aux2  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
									aux3  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
									aux0 = _mm_add_epi8(aux2,  aux3);
									_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_2+disp_3),   aux0);
								}
							}

						}

						// Copy score profiles to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_scoreProfiles[d],CL_FALSE, 0, chunk_vD[c]*SUBMAT_ROWS*sizeof(char),scoreProfiles[d],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_scoreProfiles");

						// Copy lengths to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_n[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned short int),chunk_n[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_n");

						// Copy nbbs to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_nbb[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned short int),chunk_nbb[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_nbb");

						// Copy displacement to device buffer 
						status=clEnqueueWriteBuffer(queues[d], cl_b_disp[d],CL_FALSE, 0, chunk_sequences_count[c]*sizeof(unsigned int),chunk_b_disp[c],0,NULL,NULL);
						checkErr(status,"clEnqueueWriteBuffer cl_chunk_b_disp");
					}

					// Wait for all queues to finish.
					for(d = 0; d < num_active_devices; d++) 
						clFinish(queues[d]);
					  
					for(d = 0; d < num_active_devices; d++) {

						c = current_chunk+d;

						for (int q=0; q<query_sequences_count ; q++) {

							status = clSetKernelArg(kernels[d], 1, sizeof(unsigned short int), &m[q]);
							checkError(status, "Failed to set kernel arg 1");

							status = clSetKernelArg(kernels[d], 2, sizeof(unsigned int), &a_disp[q]);
							checkError(status, "Failed to set kernel arg 2");

							scores_disp = q*chunk_sequences_count[c];
							status = clSetKernelArg(kernels[d], 7, sizeof(unsigned int), &scores_disp);
							checkError(status, "Failed to set kernel arg 7");

							// Launch the kernel
							status = clEnqueueNDRangeKernel(queues[d], kernels[d], 1, NULL, gSize, wgSize, 0, NULL, &kernel_events[d*query_sequences_count+q]);
							checkError(status, "Failed to launch kernel");
						}
					}

					// Wait for all kernels to finish.
					clWaitForEvents(num_active_devices*query_sequences_count, kernel_events);
					for(i = 0; i < num_active_devices*query_sequences_count; i++) 
						clReleaseEvent(kernel_events[i]);

					for(d = 0; d < num_active_devices; d++) {

						c = current_chunk + d;

						// Copy alignment scores to host array 
						status = clEnqueueReadBuffer(queues[d], cl_scores[d], CL_TRUE, 0, query_sequences_count*chunk_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int), tmp_scores[d], 0, NULL, NULL);
						checkErr(status,"clEnqueueReadBuffer: Couldn't read cl_scores buffer");

						// copy tmp_scores fo final scores buffer
						for (q=0; q<query_sequences_count ; q++) 
							memcpy(scores+q*vect_sequences_count*CPU_SSE_INT8_VECTOR_LENGTH+chunk_accum_sequences_count[c]*FPGA_VECTOR_LENGTH,tmp_scores[d]+(q*chunk_sequences_count[c])*FPGA_VECTOR_LENGTH,chunk_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int));
					}
				}
			}

			// CPU thread
			#pragma omp single nowait
			{
				// CPU thread: generate new parallel region
				#pragma omp parallel num_threads(cpu_threads)
				{

					__m128i  *row, *maxCol, *maxRow, *lastCol, * ptr_scores, *tmp;
					char * ptr_a, * ptr_b, * scoreProfile, *ptr_scoreProfile;

					__declspec(align(32)) __m128i score, previous, current, auxBlosum[2], auxLastCol, b_values;
					__declspec(align(32)) __m128i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
					__declspec(align(32)) __m128i vextend_gap_epi8 = _mm_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm_set1_epi8(open_gap+extend_gap), vzero_epi8 = _mm_set1_epi8(0);
					__declspec(align(32)) __m128i vextend_gap_epi16 = _mm_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm_set1_epi16(open_gap+extend_gap), vzero_epi16 = _mm_set1_epi16(0);
					__declspec(align(32)) __m128i vextend_gap_epi32 = _mm_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm_set1_epi32(open_gap+extend_gap), vzero_epi32 = _mm_set1_epi32(0);
					__declspec(align(32)) __m128i v127 = _mm_set1_epi8(127), v32767 = _mm_set1_epi16(32767);
					__declspec(align(32)) __m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);

					unsigned  int i, j, ii, jj, k, disp_1, disp_2, disp_3, dim1, dim2, nbb;
					unsigned long int s, q; 
					int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;

					int tid = omp_get_thread_num();

					// allocate memory for auxiliary buffers
					row = rows[tid];
					maxCol = maxCols[tid];
					maxRow = maxRows[tid];
					lastCol = lastCols[tid];
					scoreProfile = SPs[tid];
							
					// calculate alignment score
					#pragma omp for schedule(dynamic) nowait
					for (s=0; s< cpu_sequences_count; s++) {

						ptr_b = b + b_disp[s];
					
						// build score profile
						disp_1 = n[s]*CPU_SSE_INT8_VECTOR_LENGTH;
						for (i=0; i< n[s] ;i++ ) {
							disp_2 = i*CPU_SSE_INT8_VECTOR_LENGTH;
							// indexes
							b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
							// indexes >= 16
							aux1 = _mm_sub_epi8(b_values, v16);
							// indexes < 16
							aux2 = _mm_cmpgt_epi8(b_values,v15);
							aux3 = _mm_and_si128(aux2,vneg32);
							aux4 = _mm_add_epi8(b_values,aux3);
							ptr_scoreProfile = scoreProfile + disp_2;
							for (j=0; j< SUBMAT_ROWS; j++) {
								disp_3 = j*disp_1;
								tmp = (__m128i *) (submat + j*SUBMAT_COLS);
								auxBlosum[0] = _mm_load_si128(tmp);
								auxBlosum[1] = _mm_load_si128(tmp+1);
								aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
								aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
								aux7 = _mm_add_epi8(aux5,  aux6);
								_mm_store_si128((__m128i*)(ptr_scoreProfile+disp_3),   aux7);
							}
						}

						// caluclate number of blocks
						nbb = ceil( (double) n[s] / (double) cpu_block_size);

						for (q=0; q<query_sequences_count; q++){

							ptr_a = a + a_disp[q];
							ptr_scores = (__m128i *) (scores + (q*vect_sequences_count+s)*CPU_SSE_INT8_VECTOR_LENGTH + fpga_sequences_count*FPGA_VECTOR_LENGTH);

							// init buffers
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi8(0);
							#pragma unroll(CPU_SSE_UNROLL_COUNT)
							for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi8(0);
								
							// set score to 0
							score = _mm_set1_epi8(0);
							// calculate a[i] displacement
							disp_1 = n[s]*CPU_SSE_INT8_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								dim1 = n[s]-k*cpu_block_size;
								dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

								// init buffers
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi8(0);
								#pragma unroll(CPU_SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi8(0);
								auxLastCol = _mm_set1_epi8(0);

								for( i = 1; i < m[q]+1; i++){
								
									// previous must start in 0
									previous = _mm_set1_epi8(0);
									// update row[0] with lastCol[i-1]
									row[0] = lastCol[i-1];
									// calculate score profile displacement
									ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1 + (k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH);
									// calculate dim2
									dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
											//calcuate the diagonal value
											current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
											// calculate current max value
											current = _mm_max_epi8(current, maxRow[i]);
											current = _mm_max_epi8(current, maxCol[j]);
											current = _mm_max_epi8(current, vzero_epi8);
											// update maxRow and maxCol
											maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
											maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
											aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
											maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
											maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
											// update max score
											score = _mm_max_epi8(score,current);
											// update row buffer
											row[j-1] = previous;
											previous = current;
										}
									}
									#pragma unroll
									for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
										//calcuate the diagonal value
										current = _mm_adds_epi8(row[j-1], _mm_load_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH)));
										// calculate current max value
										current = _mm_max_epi8(current, maxRow[i]);
										current = _mm_max_epi8(current, maxCol[j]);
										current = _mm_max_epi8(current, vzero_epi8);
										// update maxRow and maxCol
										maxRow[i] = _mm_subs_epi8(maxRow[i], vextend_gap_epi8);
										maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
										aux0 = _mm_subs_epi8(current, vopen_extend_gap_epi8);
										maxRow[i] = _mm_max_epi8(maxRow[i], aux0);
										maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
										// update max score
										score = _mm_max_epi8(score,current);
										// update row buffer
										row[j-1] = previous;
										previous = current;
									}
									// update lastCol
									lastCol[i-1] = auxLastCol;
									auxLastCol = current;
								}
							}

							// store max value
							_mm_store_si128 (ptr_scores,_mm_cvtepi8_epi32(score));
							_mm_store_si128 (ptr_scores+1,_mm_cvtepi8_epi32(_mm_srli_si128(score,4)));
							_mm_store_si128 (ptr_scores+2,_mm_cvtepi8_epi32(_mm_srli_si128(score,8)));
							_mm_store_si128 (ptr_scores+3,_mm_cvtepi8_epi32(_mm_srli_si128(score,12)));

							// overflow detection
							aux1 = _mm_cmpeq_epi8(score,v127);
							overflow_flag = _mm_test_all_zeros(aux1,v127); 

							// if overflow
							if (overflow_flag == 0){

								// detect if overflow occurred in low-half, high-half or both halves
								aux1 = _mm_cmpeq_epi8(_mm_slli_si128(score,8),v127);
								bb1_start = _mm_test_all_zeros(aux1,v127);
								aux1 = _mm_cmpeq_epi8(_mm_srli_si128(score,8),v127);
								bb1_end = 2 - _mm_test_all_zeros(aux1,v127);

								// recalculate using 16-bit signed integer precision
								for (bb1=bb1_start; bb1<bb1_end ; bb1++){

									// init buffers
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi16(0);
									#pragma unroll(CPU_SSE_UNROLL_COUNT)
									for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi16(0);
									
									// set score to 0
									score = _mm_set1_epi16(0);

									disp_2 = bb1*CPU_SSE_INT16_VECTOR_LENGTH;

									for (k=0; k < nbb; k++){

										// calculate dim1
										dim1 = n[s]-k*cpu_block_size;
										dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

										// init buffers
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi16(0);
										#pragma unroll(CPU_SSE_UNROLL_COUNT)
										for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi16(0);
										auxLastCol = _mm_set1_epi16(0);

										for( i = 1; i < m[q]+1; i++){
										
											// previous must start in 0
											previous = _mm_set1_epi16(0);
											// update row[0] with lastCol[i-1]
											row[0] = lastCol[i-1];
											// calculate score profile displacement
											ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_2 + k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH;
											// calculate dim2
											dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

											for (ii=0; ii<dim2 ; ii++) {

												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
													//calcuate the diagonal value
													current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
													// calculate current max value
													current = _mm_max_epi16(current, maxRow[i]);
													current = _mm_max_epi16(current, maxCol[j]);
													current = _mm_max_epi16(current, vzero_epi16);
													// update maxRow and maxCol
													maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
													maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
													aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
													maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
													maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
													// update row buffer
													row[j-1] = previous;
													previous = current;
													// update max score
													score = _mm_max_epi16(score,current);
												}
											}
											#pragma unroll
											for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
												//calcuate the diagonal value
												current = _mm_adds_epi16(row[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
												// calculate current max value
												current = _mm_max_epi16(current, maxRow[i]);
												current = _mm_max_epi16(current, maxCol[j]);
												current = _mm_max_epi16(current, vzero_epi16);
												// update maxRow and maxCol
												maxRow[i] = _mm_subs_epi16(maxRow[i], vextend_gap_epi16);
												maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
												aux0 = _mm_subs_epi16(current, vopen_extend_gap_epi16);
												maxRow[i] = _mm_max_epi16(maxRow[i], aux0);
												maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
												// update row buffer
												row[j-1] = previous;
												previous = current;
												// update max score
												score = _mm_max_epi16(score,current);
											}
											// update lastCol
											lastCol[i-1] = auxLastCol;
											auxLastCol = current;
										}
									}
									// store max value
									_mm_store_si128 (ptr_scores+bb1*2,_mm_cvtepi16_epi32(score));
									_mm_store_si128 (ptr_scores+bb1*2+1,_mm_cvtepi16_epi32(_mm_srli_si128(score,8)));

									// overflow detection
									aux1 = _mm_cmpeq_epi16(score,v32767);
									overflow_flag = _mm_test_all_zeros(aux1,v32767); 

									// if overflow
									if (overflow_flag == 0){

									// detect if overflow occurred in low-half, high-half or both halves
										aux1 = _mm_cmpeq_epi16(_mm_slli_si128(score,8),v32767);
										bb2_start = _mm_test_all_zeros(aux1,v32767);
										aux1 = _mm_cmpeq_epi16(_mm_srli_si128(score,8),v32767);
										bb2_end = 2 - _mm_test_all_zeros(aux1,v32767);

										// recalculate using 32-bit signed integer precision
										for (bb2=bb2_start; bb2<bb2_end ; bb2++){
	
											// init buffers
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) maxRow[i] = _mm_set1_epi32(0);
											#pragma unroll(CPU_SSE_UNROLL_COUNT)
											for (i=0; i<m[q]+1 ; i++ ) lastCol[i] = _mm_set1_epi32(0);
												
											// set score to 0
											score = _mm_set1_epi32(0);

											disp_3 = disp_2 + bb2*CPU_SSE_INT32_VECTOR_LENGTH;

											for (k=0; k < nbb; k++){
												// calculate dim1
												dim1 = n[s]-k*cpu_block_size;
												dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

												// init buffers
												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
												#pragma unroll(CPU_SSE_UNROLL_COUNT)
												for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi32(0);
												auxLastCol = _mm_set1_epi32(0);

												for( i = 1; i < m[q]+1; i++){
												
													// previous must start in 0
													previous = _mm_set1_epi32(0);
													// update row[0] with lastCol[i-1]
													row[0] = lastCol[i-1];
													// calculate score profile displacement
													ptr_scoreProfile = scoreProfile+((int)(ptr_a[i-1]))*disp_1+disp_3 + k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH;
													// calculate dim2
													dim2 = dim1 / CPU_SSE_UNROLL_COUNT;

													for (ii=0; ii<dim2 ; ii++) {

														#pragma unroll(CPU_SSE_UNROLL_COUNT)
														for( j=ii*CPU_SSE_UNROLL_COUNT+1, jj=0; jj < CPU_SSE_UNROLL_COUNT; jj++, j++) {
															//calcuate the diagonal value
															current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
															// calculate current max value
															current = _mm_max_epi32(current, maxRow[i]);
															current = _mm_max_epi32(current, maxCol[j]);
															current = _mm_max_epi32(current, vzero_epi32);
															// update maxRow and maxCol
															maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
															maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
															aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
															maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
															maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
															// update row buffer
															row[j-1] = previous;
															previous = current;
															// update max score
															score = _mm_max_epi32(score,current);
														}
													}
													#pragma unroll
													for( j = dim2*CPU_SSE_UNROLL_COUNT+1; j < dim1+1; j++) {
														//calcuate the diagonal value
														current = _mm_add_epi32(row[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile+(j-1)*CPU_SSE_INT8_VECTOR_LENGTH))));
														// calculate current max value
														current = _mm_max_epi32(current, maxRow[i]);
														current = _mm_max_epi32(current, maxCol[j]);
														current = _mm_max_epi32(current, vzero_epi32);
														// update maxRow and maxCol
														maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap_epi32);
														maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
														aux0 = _mm_sub_epi32(current, vopen_extend_gap_epi32);
														maxRow[i] = _mm_max_epi32(maxRow[i], aux0);
														maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
														// update row buffer
														row[j-1] = previous;
														previous = current;
														// update max score
														score = _mm_max_epi32(score,current);
													}
													// update lastCol
													lastCol[i-1] = auxLastCol;
													auxLastCol = current;
												}
											}
											// store max value
											_mm_store_si128 (ptr_scores+bb1*2+bb2,score);
										}
									}
								}
							}
						}
					}
				}
			}

		}

		// check overflow
		for (q=0; q< query_sequences_count; q++){

			disp_1 = q*vect_sequences_count*CPU_SSE_INT8_VECTOR_LENGTH;

			for (c=0; c< chunk_count; c++){

				disp_2 = disp_1 + chunk_accum_sequences_count[c]*FPGA_VECTOR_LENGTH;
		
				#pragma omp parallel private(i,j,ptr_scores) num_threads(cpu_threads)
				{
					int tid = omp_get_thread_num();

					#pragma omp for schedule(dynamic)
					for (s=0;s<chunk_sequences_count[c] ; s++) {

						ptr_scores = scores + disp_2 + s*FPGA_VECTOR_LENGTH;
						int overflow_flag=0, overflow[FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR] = {0};					
						
						for (i=0; i < FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR ; i++) {
							int start = i*CPU_SSE_INT32_VECTOR_LENGTH, end=(i+1)*CPU_SSE_INT32_VECTOR_LENGTH;
							for (j=start; j< end; j++)
								if (ptr_scores[j] == CHAR_MAX)
									overflow[i]++;
							overflow_flag += overflow[i];
						}
						if (overflow_flag > 0) 
							sw_host(a+a_disp[q],m[q],chunk_b[c]+chunk_b_disp[c][s]*FPGA_VECTOR_LENGTH,chunk_n[c][s],submat,ptr_scores,overflow,open_gap,extend_gap,SPs[tid],(__m128i *)rows[tid],(__m128i *)maxRows[tid],(__m128i *)maxCols[tid],(__m128i *)lastCols[tid]);
			
					}
				}
			}

		}


		workTime = dwalltime() - tick;

		// Wait for command queue to complete pending events
		for (d=0; d<num_devices ; d++)
			status = clFinish(queues[d]);
		checkError(status, "Failed to finish");

		// Free allocated memory
		free(a);
		free(a_disp);
		if (chunk_count > 0) {
			free(chunk_b);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_n[i]);
			free(chunk_n);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_nbb[i]);
			free(chunk_nbb);
			for (i=0; i< chunk_count ; i++ ) 
				free(chunk_b_disp[i]);
			free(chunk_b_disp);
			free(chunk_sequences_count);
			free(chunk_accum_sequences_count);
			free(chunk_vD);
		}
		free(b);
		free(n);
		free(b_disp);

		// Load database headers
		load_database_headers (sequences_filename, sequences_count, max_title_length, &sequence_headers);

		// Print top scores
		tmp_sequence_headers = (char**) malloc(sequences_count*sizeof(char *));
		for (i=0; i<query_sequences_count ; i++ ) {
			memcpy(tmp_sequence_headers,sequence_headers,sequences_count*sizeof(char *));
			sort_scores(scores+i*vect_sequences_count*CPU_SSE_INT8_VECTOR_LENGTH,tmp_sequence_headers,sequences_count,cpu_threads);
			printf("\nQuery no.\t\t\t%d\n",i+1);
			printf("Query description: \t\t%s\n",a_headers[i]+1);
			printf("Query length:\t\t\t%d residues\n",m[i]);
			printf("\nScore\tSequence description\n");
			for (j=0; j<top; j++) 
				printf("%d\t%s",scores[i*vect_sequences_count*CPU_SSE_INT8_VECTOR_LENGTH+j],tmp_sequence_headers[j]+1);
		}
		printf("\nSearch date:\t\t\t%s",ctime(&current_time));
		printf("Search time:\t\t\t%lf seconds\n",workTime);
		printf("Search speed:\t\t\t%.2lf GCUPS\n",(Q*D) / ((workTime+(test_fpga_time>test_cpu_time?test_fpga_time:test_cpu_time))*1000000000));
		printf("CPU threads:\t\t\t%d\n",cpu_threads);
		printf("CPU vector length:\t\t%d\n",CPU_SSE_INT8_VECTOR_LENGTH);
		printf("CPU block width:\t\t%d\n",cpu_block_size);
		printf("Number of FPGAs:\t\t%u\n",num_devices);
		printf("FPGA vector length:\t\t%d\n",FPGA_VECTOR_LENGTH);
		printf("FPGA block width:\t\t%d\n",FPGA_BLOCK_WIDTH);
		printf("Max. chunk size in FPGA:\t%ld bytes\n",max_chunk_size);

		// Free allocated memory
		free(m);
		free(scores); 	
		for (i=0; i<query_sequences_count ; i++ ) 
			free(a_headers[i]);
		free(a_headers);
		for (i=0; i<sequences_count ; i++ ) 
			free(sequence_headers[i]);
		free(sequence_headers);
		free(tmp_sequence_headers);
		for (d=0; d<num_devices ; d++) {
			free(tmp_scores[d]);
			free(scoreProfiles[d]);
		}
		for (i=0; i<cpu_threads ; i++) {
			free(rows[i]);
			free(maxRows[i]);
			free(maxCols[i]);
			free(lastCols[i]);
			free(SPs[i]);
		}
		free(rows);
		free(maxRows);
		free(maxCols);
		free(lastCols);
		free(SPs);

		// free FPGA resources
		for (d=0; d<num_devices ; d++){
			clReleaseMemObject(cl_a[d]);
			clReleaseMemObject(cl_n[d]);
			clReleaseMemObject(cl_nbb[d]);
			clReleaseMemObject(cl_b_disp[d]);
			clReleaseMemObject(cl_scoreProfiles[d]);
			clReleaseMemObject(cl_scores[d]);
		}

		// Free the resources allocated
		cleanup();


}