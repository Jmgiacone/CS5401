#!/bin/bash

# Uses /usr/bin/time to get Wall time (seconds) and Max memory consumption (KB)
# Pipes MiniSat output through Grep and Awk to get total decisions made over all CNF files.
( /usr/bin/time -f "%e\n%M" ./solvers/minisat/minisat $@ | grep -oP "decisions\s*:\s*(\K[0-9]+)" | awk '{sum += $1} END {print sum}' ) 2>&1
