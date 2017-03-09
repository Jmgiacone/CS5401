#!/bin/bash

# Place your compile and execute script here.
# You can write any bash script that will run on campus linux machines.
# The below script will compile and execute the HelloEC program.
# HelloEC will also be passed the first argument to this script, a config file.

# Compile 
g++ -g -W -Wall -pedantic-errors -std=c++11 src/main.cpp -o hw1a

time ./hw1a $1 

rm file.txt
rm output.txt
