#include "arguments.h"

const char *argp_program_bug_address =
  "<erucci@lidi.info.unlp.edu.ar>";

/* Program documentation. */
static char doc[] =
  "\nOSWALD is a software to accelerate Smith-Waterman protein database search on heterogeneous architectures based on Intel processors and Altera FPGAs";

void program_arguments_processing (int argc, char * argv[]) {

	int i, result, arg_count=4;

	struct argp_option options[] =	{
		{ 0, 0, 0, 0, "OSWALD execution", 1},
		{ 0, 'O', "<string>", 0, "'preprocess' for database preprocessing, 'search' for database search, 'info' for FPGA information [REQUIRED]", 1},
		{ 0, 0, 0, 0, "preprocess", 2},
		{ "input", 'i', "<string>", 0, "Input sequence filename (must be in FASTA format). [REQUIRED]", 2},
		{ "output", 'o', "<string>", 0, "Output filename. [REQUIRED]", 2},
		{ 0, 0, 0, 0, "search", 3},
		{ "query", 'q', "<string>", 0,  "Input query sequence filename (must be in FASTA format). [REQUIRED]", 3},
		{ "db", 'd', "<string>", 0, "Preprocessed database output filename. [REQUIRED]", 3},
		{ "sm", 's', "<string>", 0, "Substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250 (default: blosum62).", 3},
		{ "gap_open", 'g', "<integer>", 0, "Gap open penalty (default: 10).", 3},
		{ "gap_extend", 'e', "<integer>", 0, "Gap extend penalty (default: 2).", 3},
		{ "execution_mode", 'm', "<integer>", 0, "0 for FPGA mode, 1 for hybrid mode (default: 1).", 3},
		{ "cpu_threads", 'c', "<integer>", 0, "Number of CPU threads (default: 16).", 3},
		{ "vector_length", 'v', "<integer>", 0, "Vector length in host: 16 for SSE, 32 for AVX2 (default: 16).", 3},
		{ "cpu_block_width", 'b', "<integer>", 0, "CPU block width (default: 256).", 3},
		{ "num_fpgas", 'f', "<integer>", 0, "Number of FPGAs (default: 1).", 3},
		{ "max_chunk_size", 'k', "<integer>", 0, "Maximum chunk size in FPGA (bytes, default: 134217728).", 3},
		{ "db_percentage", 'p', "<integer>", 0, "Database percentage for testing computational power (default: 0.01).", 3},
		{ "top", 'r', "<integer>", 0, "Number of scores to show (default: 10).", 3},
		{ 0 }
	};

	struct argp argp = { options, parse_opt, 0, doc};
	result = argp_parse (&argp, argc, argv, 0, 0, &arg_count); 
}

static int parse_opt (int key, char *arg, struct argp_state *state) {

	int *arg_count = (int *) state->input;

	switch(key) {
		case 'O': 
			if ((strcmp(arg,"preprocess") != 0) && (strcmp(arg,"search") != 0) && (strcmp(arg,"info") != 0))
				argp_failure (state, 1, 0, "%s is not a valid option for execution.",arg);
			else
				op = arg;
			break;
		case 'i':
			input_filename = arg;
			break;
		case 'o':
			output_filename = arg;
			break;
		case 'q':
			queries_filename = arg;
			break;
		case 'd':
			sequences_filename = arg;
			break;
		case 's':
			if ((strcmp(arg,"blosum45") != 0) && (strcmp(arg,"blosum50") != 0) && (strcmp(arg,"blosum62") != 0) && (strcmp(arg,"blosum80") != 0) && (strcmp(arg,"blosum90") != 0) && (strcmp(arg,"pam30") != 0) && (strcmp(arg,"pam70") != 0) && (strcmp(arg,"pam250") != 0))
				argp_failure (state, 1, 0, "%s is not a valid option for substitution matrix.",arg);
			else {
				if (strcmp(arg,"blosum45") == 0) { submat = blosum45; strcpy(submat_name,"BLOSUM45"); }
				if (strcmp(arg,"blosum50") == 0) { submat = blosum50; strcpy(submat_name,"BLOSUM50"); }
				if (strcmp(arg,"blosum62") == 0) { submat = blosum62; strcpy(submat_name,"BLOSUM62"); }
				if (strcmp(arg,"blosum80") == 0) { submat = blosum80;  strcpy(submat_name,"BLOSUM80"); }
				if (strcmp(arg,"blosum90") == 0) { submat = blosum90; strcpy(submat_name,"BLOSUM90"); }
				if (strcmp(arg,"pam30") == 0) { submat = pam30;  strcpy(submat_name,"PAM30"); }
				if (strcmp(arg,"pam70") == 0) {	submat = pam70; strcpy(submat_name,"PAM70"); }
				if (strcmp(arg,"pam250") == 0) { submat = pam250;  strcpy(submat_name,"PAM250"); }
			}
			break;
		case 'g':
			open_gap = atoi(arg);
			if ((open_gap < 0) || (open_gap > 255))
				argp_failure (state, 1, 0, "%s is not a valid option for gap open penalty.",open_gap);
			break;
		case 'e':
			extend_gap = atoi(arg);
			if ((extend_gap < 0) || (extend_gap > 127))
				argp_failure (state, 1, 0, "%s is not a valid option for gap extend penalty.",extend_gap);
			break;
		case 'm':
			execution_mode = atoi(arg);
			if ((execution_mode != FPGA_MODE) && (execution_mode != HYBRID_MODE))
				argp_failure (state, 1, 0, "%d is not a valid option for execution mode.",execution_mode);
			break;
		case 'c':
			cpu_threads = atoi(arg);
			if (cpu_threads < 0)
				argp_failure (state, 1, 0, "The number of host threads must be greater than 0.");
			break;
		case 'b':
			cpu_block_size = atoi(arg);
			if (cpu_block_size < 0)
				argp_failure (state, 1, 0, "The host block width must be greater than 0.");
			break;
		case 'v':
			cpu_vector_length = atoi(arg);
			if ((cpu_vector_length != 16) && (cpu_vector_length != 32))
				argp_failure (state, 1, 0, "%d is not a valid option for vector length of host.",cpu_vector_length);
			break;
		case 'f':
			num_devices = atoi(arg);
			if (num_devices < 0)
				argp_failure (state, 1, 0, "The number of FPGAs must be greater than 0.");
			break;
		case 'k':
			max_chunk_size = atol(arg);
			if (max_chunk_size < 0)
				argp_failure (state, 1, 0, "The chunk size must be greater than 0.");
			break;
		case 'p':
			test_db_percentage = atof(arg);
			if ((test_db_percentage <= 0) || (test_db_percentage > 1))
				argp_failure (state, 1, 0, "The database percentage for testing must be between 0 and 1.");
			break;
		case 'r':
			top = atoi(arg);
			if (top < 0)
				argp_failure (state, 1, 0, "The number of scores to show must be greater than 0.");
			break;
		case ARGP_KEY_END:
			if (*arg_count == 1)
				argp_failure (state, 1, 0, "Missing options");
			if (op == NULL)
				argp_failure (state, 1, 0, "OSWALD execution option is required");
			else 
				if (strcmp(op,"preprocess") == 0){
					if (input_filename == NULL)
						argp_failure (state, 1, 0, "Input sequence filename is required");
					if (output_filename == NULL)
						argp_failure (state, 1, 0, "Output filename is required");
				} else {
					if (strcmp(op,"search") == 0){
						if (sequences_filename == NULL)
							argp_failure (state, 1, 0, "Database filename is required");
						if (queries_filename == NULL)
							argp_failure (state, 1, 0, "Query sequences filename is required");
					}
				}
/*	    default:
			return ARGP_ERR_UNKNOWN;
*/	}

	return 0;
} 

