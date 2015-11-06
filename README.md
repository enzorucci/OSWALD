# OSWALD
OpenCL Smith-Waterman Algorithm on Altera FPGA for Large Protein Databases

## Description
OSWALD is a software to accelerate Smith-Waterman protein database search on heterogeneous architectures based on Altera FPGAs. It exploits OpenMP multithreading and SIMD computing through SSE and AVX2 extensions on the host while it takes advantage of pipeline and vectorial parallelism on the FPGAs. 

### OSWALD main characteristics

* Portable

Altera OpenCL SDK provides portability to FPGA code.

* Completely functional

Queries, database, substitution matrix and gap penalty values are configurable.

* General

OSWALD estimates relative compute power among host and devices to reach a well-balanced workload distribution.

### OSWALD execution and performance

In addition, it offers two execution modes: (1) FPGA(s) and (2) concurrent host and FPGA(s). On a heterogeneous platform based on two Xeon E5-2695 v3 and a single Altera Stratix V GSD5 Half-Length PCIe Boards, OSWALD reaches up to 58 GCUPS on FPGA and 401 GCUPS on hybrid mode (host+FPGA), while searching Environmental NR database.

## Usage
Databases must be preprocessed before searching it.

### Parameters
OSWALD execution

      -O <string> 'preprocess' for database preprocessing, 'search' for database search, 'info' for FPGA information. [REQUIRED]

* preprocess
```
  -i,   --input=<string> Input sequence filename (must be in FASTA format). [REQUIRED]
  
  -o,   --output=<string> Output filename. [REQUIRED]
  
  -c,   --cpu_threads=<integer> Number of host threads.
```

* search
```
  -q,   --query=<string> Input query sequence filename (must be in FASTA format). [REQUIRED]
  
  -d,   --db=<string> Preprocessed database output filename. [REQUIRED]
  
  -m,   --execution_mode=<integer> Execution mode: 0 for FPGA only, 1 for concurrent host and FPGA (default: 1).
  
  -c,   --host_threads=<integer> Number of host threads (default: 4).
  
  -e,   --gap_extend=<integer> Gap extend penalty (default: 2).

  -f,   --num_fpgas=<integer> Number of FPGAs (default: 1).

  -g,   --gap_open=<integer> Gap open penalty (default: 10).

  -b,   --cpu_block_width=<integer> Host block width (default: 256).

  -k,   --max_chunk_size=<integer> Maximum chunk size in bytes (default: 134217728).

  -p,   --db_percentage=<float> Database percentage used to estimate relative compute power (default: 0.01).
  
  -r,   --top=<integer> Number of scores to show (default: 10). 
  
  -s,   --sm=<string> Substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250 (default: blosum62).
  
  -v,   --vector_length=<integer> Vector length: 16 for host with SSE support, 32 for host with AVX2 support (default: 16).
  
  -?,   --help Give this help list
        --usage Give a short usage message
```

### Examples

* Database preprocessing

  `./oswald -O preprocess -i db.fasta -o out `
  
  Preprocess *db.fasta* database using 4 host threads. The preprocessed database name will be *out*.
  
  `./oswald -O preprocess -i db.fasta -o out -c 8`
  
  Preprocess *db.fasta* database using 8 host threads. The preprocessed database name will be *out*.

* Database search


  `./oswald -O search -q query.fasta -d out -m 0 `
  
  Search query sequence *query.fasta* against *out* preprocessed database in FPGA mode with 1 accelerator and 4 host threads using SSE instruction set.

    `./oswald -O search -q query.fasta -d out -m 0 -f 2`
  
  Search query sequence *query.fasta* against *out* preprocessed database in FPGA mode with 2 accelerators and 4 host threads using SSE instruction set.

  `./oswald -O search -q query.fasta -d out -m 0 -c 16`
  
  Search query sequence *query.fasta* against *out* preprocessed database in FPGA mode with 1 accelerator and 16 host threads using SSE instruction set.
  
  `./oswald -O search -q query.fasta -d out -m 0 -c 16 -v 32`
  
  Search query sequence *query.fasta* against *out* preprocessed database in FPGA mode with 1 accelerator and 16 host threads using AVX2 instruction set.
  
  `./oswald -O search -q query.fasta -d out -m 1 `
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent host and FPGA mode with 4 host threads (SSE) and one single accelerator.
  
  `./oswald -O search -q query.fasta -d out -m 1 -f 2`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent host and FPGA mode with 4 host threads (SSE) and two accelerators.

  `./oswald -O search -q query.fasta -d out -m 1 -v 32`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent host and FPGA mode with 4 host threads (AVX2) and one single accelerator.  

  `./oswald -O search -q query.fasta -d out -m 1 -v 32 -b 128`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent host and FPGA mode with 4 host threads (AVX2, block width equal to 128) and one single accelerator.  
  
  `./oswald -O search -q query.fasta -d out -m 1 -k 67108864`
  
  Search query sequence *query.fasta* against *out* preprocessed database in concurrent host and FPGA mode with 4 host threads and one single accelerator. Divide FPGA database part in chunks of maximum size 67108864 bytes.
  
  `./oswald --help`
  
  `./oswald -?`
  
  Print help list.

### Importante notes
* Database and query files must be in FASTA format.
* Supported substitution matrixes: BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70 and PAM250.
* Workload balance and data locality exploitation are critical to achieve good performance. Tune the database percentage used to estime relative compute power and the host block width with the *-p* and *-b* options, respectively.

## Reference
*Under evaluation*

## Changelog
* November 06, 2015 (v1.0)
Binary code released

## Contact
If you have any question or suggestion, please contact Enzo Rucci (erucci [at] lidi.info.unlp.edu.ar)
