
#define SUBMAT_ROWS 23
#define VECTOR_LENGTH 16
#define FPGA_BLOCK_WIDTH 28
#define CHANNEL_MAX_DEPTH 5478

#pragma OPENCL EXTENSION cl_altera_channels : enable

channel char16 maxRows_chan __attribute__((depth(CHANNEL_MAX_DEPTH))) ;
channel char16 lastCols_chan __attribute__((depth(CHANNEL_MAX_DEPTH)));

// AOC kernel demonstrating device-side printf call
__attribute__((reqd_work_group_size(1,1,1)))
__attribute__((num_compute_units(1)))
__attribute__((task))
__kernel void sw (__global const char * restrict a, const unsigned short int m, const unsigned int a_disp,
					const char open_extend_gap, const char extend_gap,
					__global const char16 * restrict scoreProfiles, __global int16 * restrict scores, const unsigned int scores_disp,
					const unsigned int s_start, const unsigned int s_end, __global const unsigned short int * restrict b_lengths,
					__global const unsigned short int * restrict nbbs,	__global const unsigned int * restrict b_disp){

	__global const char * ptr_a = a + a_disp;
	char private_a[CHANNEL_MAX_DEPTH];

	// copy sequence a to private memory
	for(int i = 0; i < m; i++)
		private_a[i] = ptr_a[i];

	for (unsigned int s=s_start; s < s_end; s++) {

		char16 score[FPGA_BLOCK_WIDTH]={0};
		// get scoreProfile 	
		__global const char16 * scoreProfile = scoreProfiles + b_disp[s]*SUBMAT_ROWS;
		// get n
		unsigned short int n = b_lengths[s];
		// get nbb
		unsigned short int nbb = nbbs[s];

		for (int k=0; k<nbb; k++) {

			char16 row[FPGA_BLOCK_WIDTH]={0}, maxCol[FPGA_BLOCK_WIDTH]={0}, maxRow, previous, current, auxLastCol=0;
			unsigned int block_disp = k*FPGA_BLOCK_WIDTH;
			
			for(int i = 0; i < m; i++){

				// previous must start in 0
				previous = 0;
				if (k == 0){
					maxRow = 0;
				} else {
					// update row[0] with lastCols[i]
					row[0] = read_channel_altera(lastCols_chan);
					maxRow = read_channel_altera(maxRows_chan);
				}

				// copy SP to private memory
				__global const char16 * ptr_scoreProfile = scoreProfile + ((int)private_a[i])*n + block_disp;
				
				#pragma unroll
				for (int j=0; j < FPGA_BLOCK_WIDTH ; j++){
					//calcuate the diagonal value
					current = add_sat(row[j],ptr_scoreProfile[j]);
					// calculate current max value
					current = max(current, maxRow);
					current = max(current, maxCol[j]);
					current = max(current, 0);
					// update maxRow and maxCol
					char16 aux1 = maxRow - extend_gap;
					char16 aux2 = maxCol[j] - extend_gap;
					char16 aux3 = current - open_extend_gap;
					maxRow = max(aux1, aux3);
					maxCol[j] =  max(aux2, aux3);	
					// update row buffer
					row[j] = previous;
					previous = current;
					// update max score
					score[j] = max(score[j], current);
				}
				if (k != nbb-1) {
					// update lastCol
					write_channel_altera(lastCols_chan,auxLastCol);
					auxLastCol = current;
					// update maxRow
					write_channel_altera(maxRows_chan,maxRow);
				}
			}
		}
		// store block max score
		#pragma unroll
		for (int j=1; j <FPGA_BLOCK_WIDTH ; j++) 
			score[0] = max(score[0],score[j]);
		scores[scores_disp+s] = convert_int16(score[0]);
	}
}


