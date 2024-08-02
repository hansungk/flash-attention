#!/usr/bin/env python

import csv
import sys
import subprocess

def read_and_print_csv(file_path):
    combined_list = []
    try:
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if len(row) >= 4:
                    combined_list.append([row[0], row[1], row[2], row[3]])
                else:
                    print("Row has less than 4 fields")
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return combined_list


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} argsfile")
    sys.exit(1)

argfile = sys.argv[1]
arglist = read_and_print_csv(argfile)

for args in arglist:
    [batchsize, seqlen, headdim, causal] = args
    kernelname = f"flash_attn_func_bs{batchsize}_seqlen{seqlen}_headdim{headdim}_causal{causal}"

    command = ["ncu", "--config-file", "off", "--export", kernelname, \
               "--force-overwrite", \
               "--kernel-name", "regex:.*flash.*", \
               # "--launch-count", "4", \
               "--set", "full"]
    command = command + ["python", "run_flashattn.py"]
    command.append(batchsize)
    command.append(seqlen)
    command.append(headdim)
    command.append(causal)
    try:
        print(f"{sys.argv[0]}: running command: {command}")
        result = subprocess.run(command)
    except Exception as e:
        print(f"error running command: {e}")
