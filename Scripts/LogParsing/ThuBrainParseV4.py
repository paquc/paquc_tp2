#!/usr/bin/env python

import sys
sys.path.append('../../')
from logparser.Brain import LogParser

dataset    = 'THUNDERBIRD'  # The log file name

# Define the directory containing the log file to be parsed
input_dir = '../../data/Thunderbird/'  # Path to the directory containing the log file

# Define the directory where the parsing results will be saved
output_dir = '../../data/Thunderbird/Thunderbird_Brain_results/Thunderbird_Brain_results_V4/'  # Output directory for the parsed results

# Specify the log file to be parsed
log_file = 'Thunderbird_5M.log'  # The name of the input log file
log_format = '<AlertFlagLabel> <EpochTime> <Date> <Noeud> <Month> <Day> <Hour> <Content>'
#log_format = '<AlertFlagLabel> <EpochTime> <Date> <Noeud> <Month> <Day> <Hour> <NodeFull> <Process>: <Content>'
#log_format = '<AlertFlagLabel> <EpochTime> <Date> <NodeLoc> <FullDateTime> <NodeLocSecond> <Type> <SubSys> <Severity> <Content>'

# Regular expression list for optional preprocessing (default: [])
regex = [
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]

threshold  = 2      # Similarity threshold
delimeter  = []     # Depth of all leaf nodes

print(f"Parsing started: {log_file}")

parser = LogParser(logname=dataset, log_format=log_format, indir=input_dir, 
                   outdir=output_dir, threshold=threshold, delimeter=delimeter, rex=regex)

parser.parse(log_file)

print(f"Parsing done: {log_file}")
