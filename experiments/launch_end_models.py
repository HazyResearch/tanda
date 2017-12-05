from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import numpy as np
import os
import subprocess
import time

from .utils import get_log_dir_path, num_procs_open
from collections import OrderedDict
from itertools import product


SLEEP = 10
CTR_LIM = 12


# The args that we need from the TAN experiment config file
REQUIRED_ARGS = [
    'generator', 'gen_config', 'run_index', 'seq_len', 'tan', 'tfs'
]


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Launch end models using already-trained TAN checkpoints')
    parser.add_argument('--script', type=str,
                        help='Path to the experiment run script.')
    parser.add_argument('--end_model_config', type=str,
                        help='Path to the experiment config.')
    parser.add_argument('--tan_log_root', type=str,
        help="""Path to the experiment log directory, e.g.
        'experiments/log/2017_04_11/tan_only_16_12_32'""")
    parser.add_argument('--model_indexes', type=int, nargs='+',
                        help='List of TAN run indexes to train end model with')
    parser.add_argument('--procs_lim', type=int, default=10,
                        help='Max number of processes to run in parallel.')
    args = parser.parse_args()
    print(args)

    # Load end model config file
    with open(args.end_model_config, 'rb') as f:
        end_model_config = json.load(f, object_pairs_hook=OrderedDict)
        end_model_param_dict = end_model_config.get('parameters')
        name = end_model_config.get('name')

    # Iterate over the models to run
    run_idxs = []
    procs    = []
    j = -1
    for i in args.model_indexes:

        # Load TAN config file + add config to it
        fp = os.path.join(args.tan_log_root, 'tan', str(i), 'logs/run_log.json')
        print("\nTraining model #%s from %s..." % (i, fp))
        with open(fp, 'rb') as f:
            tan_param_dict = json.load(f, object_pairs_hook=OrderedDict)
        
        # Form the merged config dict
        param_dict = {}
        for k in REQUIRED_ARGS:
            param_dict[k] = [tan_param_dict[k]]

        # Also hardcode end_model_only=True and set log_path_tan
        param_dict['run_type'] = ['tanda-pretrained']
        param_dict['tan_checkpoint_path'] = [
            os.path.join(args.tan_log_root, 'tan', str(i), 'checkpoints')
        ]

        # Overwite everything with the end model config
        param_dict.update(end_model_param_dict)
        
        param_names = param_dict.keys()
        param_vals = [param_dict[k] for k in param_names]

        # Fix datetime here so all runs get same one
        log_path = get_log_dir_path(args.tan_log_root, name)
        print("Log path:", log_path)
    
        # Iterate over the param configs and launch subprocesses
        for param_set in product(*param_vals):
            j += 1
            run_idxs.append(j)

            # Assemble command line argument
            proc_args = [
                'python', args.script, 
                '--run_name', "%s_model_%s" % (name, i),
                '--run_index', str(j),
                '--log_path', log_path
            ]
            for k, v in zip(param_names, param_set):
                if k not in ['run_name', 'run_index', 'log_path']:
                    proc_args += ['--{0}'.format(k), str(v)]

            # Launch as subprocess
            print("Launching model {0}".format(j))
            print("\t".join(
                ["%s=%s" % (k, v) for k, v in zip(param_names, param_set)]
            ))
            p = subprocess.Popen(proc_args)
            procs.append(p)
            ctr = 0
            while True:
                k = num_procs_open(procs)
                ctr += 1
                if ctr >= CTR_LIM:
                    ctr = 0
                    print('{0} processes still running'.format(k))
                if num_procs_open(procs) >= args.procs_lim:
                    time.sleep(SLEEP)
                else:
                    break

    n = len(procs)
    ctr = 0
    while True:
        k = num_procs_open(procs)
        if k == 0:
            break
        ctr += 1
        if ctr >= CTR_LIM:
            ctr = 0
            print('{0} processes still running'.format(k))
        time.sleep(SLEEP)
