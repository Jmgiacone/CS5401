#!/bin/bash

# Place your compile and execute script here.
# You can write any bash script that will run on campus linux machines.
# The below script will compile and execute the HelloEC program.
# HelloEC will also be passed the first argument to this script, a config file.

# Compile 
g++ -g -W -Wall -pedantic-errors -std=c++11 -lboost_program_options src/main.cpp -o hw1c

time ./hw1c $1
