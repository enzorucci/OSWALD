#include "FPGAsearch.h"

// SW search using FPGA
void fpga_search() {

	char * a, ** a_headers, **chunk_b, ** tmp_sequence_headers, **sequence_headers, * scoreProfiles[MAX_NUM_DEVICES], *ptr_scoreProfile, *ptr_b;
	unsigned short int * m, sequences_db_max_length, ** chunk_n, n, **chunk_nbbs, query_sequences_max_length, *tmp_n, *tmp_nbb;
	unsigned int * a_disp, chunk_count, * chunk_vect_sequences_count, ** chunk_b_disp, **kernel_chunk_b_disp, score_pos, *tmp_disp, s_start=0, scores_disp;
	int max_title_length, open_extend_gap, num_active_devices;
	int * scores, *tmp_scores[MAX_NUM_DEVICES], *ptr_scores;
	unsigned long int query_sequences_count, Q, sequences_count, D, max_chunk_vD, * chunk_vect_accum_sequences_count, * chunk_vD;
	unsigned long int vect_sequences_count, vD, i, j, k, q, c, s, d, disp_1, disp_2, disp_3;
    time_t current_time = time(NULL);
	double workTime, tick;
	// CL vars
	cl_int status;
	cl_mem cl_a[MAX_NUM_DEVICES], cl_scores[MAX_NUM_DEVICES], cl_scoreProfiles[MAX_NUM_DEVICES], cl_n[MAX_NUM_DEVICES];
	cl_mem cl_nbb[MAX_NUM_DEVICES], cl_b_disp[MAX_NUM_DEVICES];
	// SSE vars
	__m128i ** rows, ** maxCols, ** maxRows, ** lastCols;
	char ** SPs;
	 // Configure work set over which the kernel will execute
	size_t wgSize[3] = {1, 1, 1};
	size_t gSize[3] = {1, 1, 1};

		// Print database search information
		printf("\nOSWALD v%s \n\n",VERSION);
		printf("Database file:\t\t\t%s\n",sequences_filename);

		// Load query sequences
		load_query_sequences(queries_filename,&a,&a_headers,&m,&query_sequences_count,&Q,&a_disp,cpu_threads);

		// Load db sequences and assemble chunks
		assemble_multiple_chunks_db (sequences_filename, FPGA_VECTOR_LENGTH, max_chunk_size, num_devices, &sequences_count, &D, &sequences_db_max_length, &max_title_length,
			&vect_sequences_count, &vD, &chunk_b, &chunk_count, &chunk_vect_sequences_count, &chunk_vect_accum_sequences_count, &chunk_vD, &max_chunk_vD, &chunk_n, &chunk_nbbs, &chunk_b_disp, cpu_threads);

		// update disp array to avoid division operation in kernel
		posix_memalign((void**)&kernel_chunk_b_disp, AOCL_ALIGNMENT, chunk_count*sizeof(unsigned int *));
		for (c=0; c < chunk_count ; c++) {
			posix_memalign((void**)&kernel_chunk_b_disp[c], AOCL_ALIGNMENT, chunk_vect_sequences_count[c]*sizeof(unsigned int));
			for (s=0; s<chunk_vect_sequences_count[c] ; s++) 
				kernel_chunk_b_disp[c][s] = chunk_b_disp[c][s] / FPGA_VECTOR_LENGTH;
		}
		
		// alloca memory for 32-bit computing
		posix_memalign((void**)&rows, 32, cpu_threads*sizeof(int *));
		posix_memalign((void**)&maxCols, 32, cpu_threads*sizeof(int *));
		posix_memalign((void**)&maxRows, 32, cpu_threads*sizeof(int *));
		posix_memalign((void**)&lastCols, 32, cpu_threads*sizeof(int *));
		posix_memalign((void**)&SPs, 32, cpu_threads*sizeof(char *));
		for (i=0; i<cpu_threads ; i++){
			posix_memalign((void**)&rows[i], 32, (cpu_block_size+1)*sizeof(__m128i));
			posix_memalign((void**)&maxCols[i], 32, (cpu_block_size+1)*sizeof(__m128i));
			posix_memalign((void**)&maxRows[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m128i));
			posix_memalign((void**)&lastCols[i], 32, (m[query_sequences_count-1]+1)*sizeof(__m128i));
			posix_memalign((void**)&SPs[i], 32, chunk_n[chunk_count-1][chunk_vect_sequences_count[chunk_count-1]-1]*SUBMAT_ROWS*FPGA_VECTOR_LENGTH*sizeof(char));
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
		posix_memalign((void**)&scores, AOCL_ALIGNMENT, (query_sequences_count*vect_sequences_count*FPGA_VECTOR_LENGTH)*sizeof(int));
		tmp_sequence_headers = (char**) malloc(sequences_count*sizeof(char *));
		for (d=0; d<num_devices ; d++) {
			posix_memalign((void**)&tmp_scores[d], AOCL_ALIGNMENT, (query_sequences_count*chunk_vect_sequences_count[0]*FPGA_VECTOR_LENGTH)*sizeof(int));
			posix_memalign((void**)&scoreProfiles[d], AOCL_ALIGNMENT, max_chunk_vD*SUBMAT_ROWS*sizeof(char));
		}

		// Allow nested parallelism
		omp_set_nested(1);

	
		tick = dwalltime();

		for (d=0; d<num_devices ; d++){

			// Create buffers in device 
			cl_a[d] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Q* sizeof(char), a, &status);
			checkErr(status,"clCreateBuffer cl_a");
			cl_n[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_vect_sequences_count[0]*FPGA_VECTOR_LENGTH*sizeof(unsigned short int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_n");
			cl_nbb[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_vect_sequences_count[0]*FPGA_VECTOR_LENGTH*sizeof(unsigned short int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_nbb");
			cl_b_disp[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, chunk_vect_sequences_count[0]*FPGA_VECTOR_LENGTH*sizeof(unsigned int), NULL, &status);
			checkErr(status,"clCreateBuffer cl_b_disp");
			cl_scoreProfiles[d] = clCreateBuffer(context, CL_MEM_READ_ONLY, max_chunk_vD*SUBMAT_ROWS, NULL, &status); // se puede optimizar: quizás el requerimiento es menor
			checkErr(status,"clCreateBuffer cl_scoreProfiles");
			cl_scores[d] = clCreateBuffer(context, CL_MEM_READ_WRITE, query_sequences_count*chunk_vect_sequences_count[0]*FPGA_VECTOR_LENGTH*sizeof(int), NULL, &status);
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


		for (k=0; k < chunk_count ; k+=num_devices) {

			num_active_devices = (num_devices < chunk_count-k ? num_devices : chunk_count-k);

			for (d=0; d<num_active_devices ; d++){

				c = k + d;

				status = clSetKernelArg(kernels[d], 9, sizeof(unsigned int), &chunk_vect_sequences_count[c]);
				checkError(status, "Failed to set kernel arg 9");

				#pragma omp parallel for shared(d,c,chunk_n,chunk_b,chunk_b_disp,submat,scoreProfiles,chunk_vect_sequences_count) default(none) num_threads(cpu_threads) schedule(dynamic) 
				for (unsigned int s=0;s<chunk_vect_sequences_count[c] ; s++) {

					// SSE vars
					__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
					__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;

					char * ptr_b = chunk_b[c] + chunk_b_disp[c][s];
					char * ptr_scoreProfile = scoreProfiles[d] + (chunk_b_disp[c][s]*SUBMAT_ROWS);

					// build score profile
					unsigned int disp_1 = chunk_n[c][s]*FPGA_VECTOR_LENGTH;
					for (int i=0; i< chunk_n[c][s] ;i++ ) {
						unsigned int disp_2 = i*FPGA_VECTOR_LENGTH;
						// indexes
						b_values = _mm_loadu_si128((__m128i *) (ptr_b + disp_2));
						// indexes >= 16
						aux1 = _mm_sub_epi8(b_values, v16);
						// indexes < 16
						aux2 = _mm_cmpgt_epi8(b_values,v15);
						aux3 = _mm_and_si128(aux2,vneg32);
						aux4 = _mm_add_epi8(b_values,aux3);
						for (int j=0; j< SUBMAT_ROWS; j++) {
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
				status=clEnqueueWriteBuffer(queues[d], cl_n[d],CL_FALSE, 0, chunk_vect_sequences_count[c]*sizeof(unsigned short int),chunk_n[c],0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_n");

				// Copy nbbs to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_nbb[d],CL_FALSE, 0, chunk_vect_sequences_count[c]*sizeof(unsigned short int),chunk_nbbs[c],0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_nbb");

				// Copy displacement to device buffer 
				status=clEnqueueWriteBuffer(queues[d], cl_b_disp[d],CL_FALSE, 0, chunk_vect_sequences_count[c]*sizeof(unsigned int),kernel_chunk_b_disp[c],0,NULL,NULL);
				checkErr(status,"clEnqueueWriteBuffer cl_chunk_b_disp");
			}

			// Wait for all queues to finish.
			for(d = 0; d < num_active_devices; d++) 
				clFinish(queues[d]);
			  
			for(d = 0; d < num_active_devices; d++) {

				c = k+d;

				for (int q=0; q<query_sequences_count ; q++) {

					status = clSetKernelArg(kernels[d], 1, sizeof(unsigned short int), &m[q]);
					checkError(status, "Failed to set kernel arg 1");

					status = clSetKernelArg(kernels[d], 2, sizeof(unsigned int), &a_disp[q]);
					checkError(status, "Failed to set kernel arg 2");

					scores_disp = q*chunk_vect_sequences_count[c];
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

				c = k + d;

				// Copy alignment scores to host array 
				status = clEnqueueReadBuffer(queues[d], cl_scores[d], CL_TRUE, 0, query_sequences_count*chunk_vect_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int), tmp_scores[d], 0, NULL, NULL);
				checkErr(status,"clEnqueueReadBuffer: Couldn't read cl_scores buffer");

				// copy tmp_scores fo final scores buffer
				for (int q=0; q<query_sequences_count ; q++) 
					memcpy(scores+(q*vect_sequences_count+chunk_vect_accum_sequences_count[c])*FPGA_VECTOR_LENGTH,tmp_scores[d]+q*chunk_vect_sequences_count[c]*FPGA_VECTOR_LENGTH,chunk_vect_sequences_count[c]*FPGA_VECTOR_LENGTH*sizeof(int));
			}
		}

		// check overflow
		for (q=0; q< query_sequences_count; q++){

			disp_1 = q*vect_sequences_count*FPGA_VECTOR_LENGTH;

			for (c=0; c< chunk_count; c++){

				disp_2 = disp_1 + chunk_vect_accum_sequences_count[c]*FPGA_VECTOR_LENGTH;
		
				#pragma omp parallel private(i,j,ptr_scores) num_threads(cpu_threads)
				{
					int tid = omp_get_thread_num();

					#pragma omp for schedule(dynamic)
					for (s=0;s<chunk_vect_sequences_count[c] ; s++) {

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
							sw_host(a+a_disp[q],m[q],chunk_b[c]+chunk_b_disp[c][s],chunk_n[c][s],submat,ptr_scores,overflow,open_gap,extend_gap,SPs[tid],rows[tid],maxRows[tid],maxCols[tid],lastCols[tid]);
			
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
		for (i=0; i< chunk_count ; i++ ) 
			free(chunk_b[i]);
		free(chunk_b);
		for (i=0; i< chunk_count ; i++ ) 
			free(chunk_n[i]);
		free(chunk_n);
		for (i=0; i< chunk_count ; i++ ) 
			free(chunk_nbbs[i]);
		free(chunk_nbbs);
		for (i=0; i< chunk_count ; i++ ) 
			free(chunk_b_disp[i]);
		free(chunk_b_disp);
		for (i=0; i< chunk_count ; i++ ) 
			free(kernel_chunk_b_disp[i]);
		free(kernel_chunk_b_disp);
		free(chunk_vect_sequences_count);
		free(chunk_vect_accum_sequences_count);
		free(chunk_vD);

		// Load database headers
		load_database_headers (sequences_filename, sequences_count, max_title_length, &sequence_headers);

		// allow nested paralelism
		//omp_set_nested(1);

		// Print top scores
		for (i=0; i<query_sequences_count ; i++ ) {
			memcpy(tmp_sequence_headers,sequence_headers,sequences_count*sizeof(char *));
			sort_scores(scores+i*vect_sequences_count*FPGA_VECTOR_LENGTH,tmp_sequence_headers,sequences_count,cpu_threads);
			printf("\nQuery no.\t\t\t%d\n",i+1);
			printf("Query description: \t\t%s\n",a_headers[i]+1);
			printf("Query length:\t\t\t%d residues\n",m[i]);
			printf("\nScore\tSequence description\n");
			for (j=0; j<top; j++) 
				printf("%d\t%s",scores[i*vect_sequences_count*FPGA_VECTOR_LENGTH+j],tmp_sequence_headers[j]+1);
		}
		printf("\nSearch date:\t\t\t%s",ctime(&current_time));
		printf("Search time:\t\t\t%lf seconds\n",workTime);
		printf("Search speed:\t\t\t%.2lf GCUPS\n",(Q*D) / (workTime*1000000000));
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

// Compute SW alignment in host using SSE instructions
void sw_host (char * a, unsigned short int m, char *b, unsigned short int n, char * submat, int * scores, int * overflow,
				char open_gap, char extend_gap, char * scoreProfile, __m128i * row,  __m128i * maxRow,  __m128i * maxCol, __m128i * lastCol) {

	// SSE vars
	__m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
	__m128i auxBlosum[2] __attribute__ ((aligned (32))), b_values, aux0, aux1, aux2, aux3, aux4, *tmp;
	__m128i score, auxLastCol, current, previous;
	__m128i vextend_gap = _mm_set1_epi32(extend_gap), vopen_extend_gap = _mm_set1_epi32(open_gap+extend_gap), vzero = _mm_set1_epi32(0);
	unsigned int disp_1, disp_2, disp_3, i, ii, j, jj, k, s, dim1, dim2;
	char * ptr_scoreProfile;
	unsigned short int nbb = ceil((double) n / (double) cpu_block_size);
	
	// build score profile
	disp_1 = n*CPU_SSE_INT8_VECTOR_LENGTH;
	for (i=0; i< n ;i++ ) {
		disp_2 = i*CPU_SSE_INT8_VECTOR_LENGTH;
		// indexes
		b_values = _mm_loadu_si128((__m128i *) (b + disp_2));
		// indexes >= 16
		aux1 = _mm_sub_epi8(b_values, v16);
		// indexes < 16
		aux2 = _mm_cmpgt_epi8(b_values,v15);
		aux3 = _mm_and_si128(aux2,vneg32);
		aux4 = _mm_add_epi8(b_values,aux3);
		for (j=0; j< SUBMAT_ROWS; j++) {
			disp_3 = j*disp_1;
			tmp = (__m128i *) (submat + j*SUBMAT_COLS);
			auxBlosum[0] = _mm_load_si128(tmp);
			auxBlosum[1] = _mm_load_si128(tmp+1);
			aux2  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
			aux3  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
			aux0 = _mm_add_epi8(aux2,  aux3);
			_mm_store_si128((__m128i*)(scoreProfile+disp_2+disp_3),   aux0);
		}
	}

	for (s=0; s< FPGA_TO_CPU_SSE_INT32_VECTOR_LENGTH_ADAPT_FACTOR ; s++){

		if (overflow[s] > 0) {
		
			// init buffers
			#pragma unroll(CPU_SSE_UNROLL_COUNT)
			for (i=0; i<m+1 ; i++ ) maxRow[i] = _mm_set1_epi32(0);
			#pragma unroll(CPU_SSE_UNROLL_COUNT)
			for (i=0; i<m+1 ; i++ ) lastCol[i] = _mm_set1_epi32(0);
								
			// set score to 0
			score = _mm_set1_epi32(0);

			disp_1 = n*CPU_SSE_INT8_VECTOR_LENGTH;
			disp_2 = s*CPU_SSE_INT32_VECTOR_LENGTH;

			for (k=0; k < nbb; k++){

				// calculate dim1
				dim1 = n-k*cpu_block_size;
				dim1 = (cpu_block_size < dim1 ? cpu_block_size : dim1);

				// init buffers
				#pragma unroll(CPU_SSE_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
				#pragma unroll(CPU_SSE_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) row[i] = _mm_set1_epi32(0);
				auxLastCol = _mm_set1_epi32(0);

				for( i = 1; i < m+1; i++){
								
					// previous must start in 0
					previous = _mm_set1_epi32(0);
					// update row[0] with lastCol[i-1]
					row[0] = lastCol[i-1];
					// calculate score profile displacement
					ptr_scoreProfile = scoreProfile+((int)(a[i-1]))*disp_1+ k*cpu_block_size*CPU_SSE_INT8_VECTOR_LENGTH + disp_2;
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
							current = _mm_max_epi32(current, vzero);
							// update maxRow and maxCol
							maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap);
							maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap);
							aux0 = _mm_sub_epi32(current, vopen_extend_gap);
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
						current = _mm_max_epi32(current, vzero);
						// update maxRow and maxCol
						maxRow[i] = _mm_sub_epi32(maxRow[i], vextend_gap);
						maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap);
						aux0 = _mm_sub_epi32(current, vopen_extend_gap);
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
			_mm_storeu_si128 ((__m128i *)(scores+s*CPU_SSE_INT32_VECTOR_LENGTH),score);
		}
	}

}