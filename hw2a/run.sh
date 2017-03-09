#!/bin/bash

# Place your compile and execute script here.
# You can write any bash script that will run on campus linux machines.
# The below script will compile and execute the HelloEC program.
# HelloEC will also be passed the first argument to this script, a config file.
if [ $# -eq 0 ]
  then
    echo "No arguments provided. Running with default.cfg"	   
    python3 src/main.py configurations/default.cfg
  else	  
    python3 src/main.py $1	  
fi
