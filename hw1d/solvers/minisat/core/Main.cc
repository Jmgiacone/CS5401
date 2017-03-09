/*****************************************************************************************[Main.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#include <errno.h>

#include <signal.h>
#include <zlib.h>

#include <dirent.h> // to get a list of files from a given directory

#include "utils/System.h"
#include "utils/ParseUtils.h"
#include "utils/Options.h"
#include "core/Dimacs.h"
#include "core/Solver.h"

using namespace Minisat;

//=================================================================================================


void printStats(Solver& solver)
{
    double cpu_time = cpuTime();
    double mem_used = memUsedPeak();
    printf("restarts              : %"PRIu64"\n", solver.starts);
    printf("conflicts             : %-12"PRIu64"   (%.0f /sec)\n", solver.conflicts   , solver.conflicts   /cpu_time);
    printf("decisions             : %-12"PRIu64"   (%4.2f %% random) (%.0f /sec)\n", solver.decisions, (float)solver.rnd_decisions*100 / (float)solver.decisions, solver.decisions   /cpu_time);
    printf("propagations          : %-12"PRIu64"   (%.0f /sec)\n", solver.propagations, solver.propagations/cpu_time);
    printf("conflict literals     : %-12"PRIu64"   (%4.2f %% deleted)\n", solver.tot_literals, (solver.max_literals - solver.tot_literals)*100 / (double)solver.max_literals);
    if (mem_used != 0) printf("Memory used           : %.2f MB\n", mem_used);
    printf("CPU time              : %g s\n", cpu_time);
}


static Solver* solver;
// Terminate by notifying the solver and back out gracefully. This is mainly to have a test-case
// for this feature of the Solver as it may take longer than an immediate call to '_exit()'.
static void SIGINT_interrupt(int signum) { solver->interrupt(); }

// Note that '_exit()' rather than 'exit()' has to be used. The reason is that 'exit()' calls
// destructors and may cause deadlocks if a malloc/free function happens to be running (these
// functions are guarded by locks for multithreaded use).
static void SIGINT_exit(int signum) {
    printf("\n"); printf("*** INTERRUPTED ***\n");
    if (solver->verbosity > 0){
        printStats(*solver);
        printf("\n"); printf("*** INTERRUPTED ***\n"); }
    _exit(1); }


lbool run_sat_instance(const char *input_file, const char* output_file, Solver &S)
{
    gzFile in = (input_file == NULL) ? gzdopen(0, "rb") : gzopen(input_file, "rb");
    if (in == NULL)
    {
       printf("ERROR! Could not open file: %s\n", input_file == NULL ? "<stdin>" : input_file), exit(1);
    }

    if (S.verbosity > 0)
    {
       printf("============================[ Problem Statistics ]=============================\n");
       printf("|                                                                             |\n");
    }
    double initial_time = cpuTime();
    parse_DIMACS(in, S);
    gzclose(in);
    FILE* res = (output_file != NULL) ? fopen(output_file, "wb") : NULL;

    if (S.verbosity > 0)
    {
       printf("|  Number of variables:  %12d                                         |\n", S.nVars());
       printf("|  Number of clauses:    %12d                                         |\n", S.nClauses());
    }

    double parsed_time = cpuTime();
    if (S.verbosity > 0)
    {
       printf("|  Parse time:           %12.2f s                                       |\n", parsed_time - initial_time);
       printf("|                                                                             |\n");
    }

    // Change to signal-handlers that will only notify the solver and allow it to terminate
    // voluntarily:
    signal(SIGINT, SIGINT_interrupt);
    signal(SIGXCPU,SIGINT_interrupt);

    if (!S.simplify())
    {
       if (res != NULL) fprintf(res, "UNSAT\n"), fclose(res);
       if (S.verbosity > 0)
       {
          printf("===============================================================================\n");
          printf("Solved by unit propagation\n");
          printStats(S);
          printf("\n");
       }
       printf("UNSATISFIABLE\n");
       return l_False;
    }

    vec<Lit> dummy;
    lbool ret = S.solveLimited(dummy);
    if (S.verbosity > 0)
    {
        printStats(S);
        printf("\n");
    }
    printf(ret == l_True ? "SATISFIABLE\n" : ret == l_False ? "UNSATISFIABLE\n" : "INDETERMINATE\n");
    if (res != NULL)
    {
       if (ret == l_True)
       {
          fprintf(res, "SAT\n");
          for (int i = 0; i < S.nVars(); i++)
          {
              if (S.model[i] != l_Undef)
              {
                  fprintf(res, "%s%s%d", (i==0)?"":" ", (S.model[i]==l_True)?"":"-", i+1);
              }
          }
          fprintf(res, " 0\n");
       }
       else if (ret == l_False)
       {
          fprintf(res, "UNSAT\n");
       }
       else
       {
          fprintf(res, "INDET\n");
       }
       fclose(res);
    }

    return ret;
}

//=================================================================================================
// Main:


int main(int argc, char** argv)
{
    try {
        setUsageHelp("USAGE: %s [options] <input-file or input-directory> <result-output-file>\n\n  where input may be either in plain or gzipped DIMACS.\n");
        // printf("This is MiniSat 2.0 beta\n");
        
#if defined(__linux__)
        fpu_control_t oldcw, newcw;
        _FPU_GETCW(oldcw); newcw = (oldcw & ~_FPU_EXTENDED) | _FPU_DOUBLE; _FPU_SETCW(newcw);
        printf("WARNING: for repeatability, setting FPU to use double precision\n");
#endif
        // Extra options:
        //
        IntOption    verb   ("MAIN", "verb",   "Verbosity level (0=silent, 1=some, 2=more).", 1, IntRange(0, 2));
        IntOption    dec_lim("MAIN", "dec-lim","Limit on number of decisions allow.\n", INT32_MAX, IntRange(0, INT32_MAX));
        IntOption    cpu_lim("MAIN", "cpu-lim","Limit on CPU time allowed in seconds.\n", INT32_MAX, IntRange(0, INT32_MAX));
        IntOption    mem_lim("MAIN", "mem-lim","Limit on memory usage in megabytes.\n", INT32_MAX, IntRange(0, INT32_MAX));
        
        parseOptions(argc, argv, true);

        Solver S;
        double initial_time = cpuTime();

        S.verbosity = verb;
        
        solver = &S;
        // Use signal handlers that forcibly quit until the solver will be able to respond to
        // interrupts:
        signal(SIGINT, SIGINT_exit);
        signal(SIGXCPU,SIGINT_exit);

        // Set limit on CPU-time:
        if (cpu_lim != INT32_MAX){
            rlimit rl;
            getrlimit(RLIMIT_CPU, &rl);
            if (rl.rlim_max == RLIM_INFINITY || (rlim_t)cpu_lim < rl.rlim_max){
                rl.rlim_cur = cpu_lim;
                if (setrlimit(RLIMIT_CPU, &rl) == -1)
                    printf("WARNING! Could not set resource limit: CPU-time.\n");
            } }

        // Set limit on virtual memory:
        if (mem_lim != INT32_MAX){
            rlim_t new_mem_lim = (rlim_t)mem_lim * 1024*1024;
            rlimit rl;
            getrlimit(RLIMIT_AS, &rl);
            if (rl.rlim_max == RLIM_INFINITY || new_mem_lim < rl.rlim_max){
                rl.rlim_cur = new_mem_lim;
                if (setrlimit(RLIMIT_AS, &rl) == -1)
                    printf("WARNING! Could not set resource limit: Virtual memory.\n");
            } }
        if (dec_lim != INT32_MAX){
           S.setDecsBudget(dec_lim);
            }
        if (argc == 1)
            printf("Reading from standard input... Use '--help' for help.\n");
 
        char* output_file = (argc >= 3) ? argv[2] : NULL;

        // check if this is a directory
        DIR *dir;
        struct dirent *next_sat;
        dir = opendir(argv[1]);
        if (dir != NULL)
        {
           while ((next_sat=readdir(dir)) != NULL)
           {
              char input_file[2048];
              strcpy(input_file, argv[1]);
              strcat(input_file, next_sat->d_name);
              run_sat_instance(input_file, output_file, S);
              S.Reset();
           }
           closedir(dir);
        }
        else
        {
           lbool ret = run_sat_instance((argc > 1) ? argv[1] : NULL, output_file, S);
#ifdef NDEBUG
           exit(ret == l_True ? 10 : ret == l_False ? 20 : 0);     // (faster than "return", which will invoke the destructor for 'Solver')
#else
           return (ret == l_True ? 10 : ret == l_False ? 20 : 0);
#endif
        }
    } catch (OutOfMemoryException&){
        printf("===============================================================================\n");
        printf("INDETERMINATE\n");
        exit(0);
    }
}
