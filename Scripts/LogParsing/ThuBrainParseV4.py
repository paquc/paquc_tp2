#!/usr/bin/env python

import sys
sys.path.append('../../')
from logparser.Brain import LogParser

if len(sys.argv) > 1:
    log_file = sys.argv[1]
else:
    print("Usage: python ThuBrainParseV4.py <log_file>")
    sys.exit(1)

dataset = 'THUNDERBIRD'  # The log file name

# Define the directory containing the log file to be parsed
input_dir = '../../data/Thunderbird/'  # Path to the directory containing the log file

# Define the directory where the parsing results will be saved
output_dir = '../../data/Thunderbird/Thunderbird_Brain_results/Thunderbird_Brain_results/50M'  # Output directory for the parsed results

# <Content> = <NodeFull> <Process>: <Content>
log_format = '<AlertFlagLabel> <EpochTime> <Date> <Noeud> <Month> <Day> <Hour> <Content>'
#log_format = '<AlertFlagLabel> <EpochTime> <Date> <Noeud> <Month> <Day> <Hour> <NodeFull> <Process>: <Content>'

# Regular expression list for optional preprocessing (default: [])
regex = []

threshold  = 2      # Similarity threshold
delimeter  = []     # Depth of all leaf nodes

print(f"Parsing started: {log_file}")

parser = LogParser(logname=dataset, log_format=log_format, indir=input_dir, 
                   outdir=output_dir, threshold=threshold, delimeter=delimeter, rex=regex)

parser.parse(log_file)

print(f"Parsing done: {log_file}")
