from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import numpy as np
import subprocess
import time

from .utils import create_config_str, get_log_dir_path, num_procs_open
from collections import OrderedDict
from itertools import product


SLEEP = 10
CTR_LIM = 12


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch TANDA experiments')
    parser.add_argument('--script', type=str,
                        help='Path to the experiment run script.')
    parser.add_argument('--config', type=str,
                        help='Path to the experiment config file.')
    parser.add_argument('--n_models', type=int, default=0,
                        help='Expected number of models.')
    parser.add_argument('--procs_lim', type=int, default=10,
                        help='Max number of processes to run in parallel.')
    parser.add_argument('--log_root', type=str, default="experiments/log",
                        help='Root for all log files.')
    args = parser.parse_args()
    print(args)

    # Load config file
    with open(args.config, 'rb') as f:
        config = json.load(f, object_pairs_hook=OrderedDict)
    
    # String in JSON config
    name = config.get('name')
    
    # Dictionary of lists in JSON config
    param_dict = config.get('parameters')
    if 'generator' in param_dict:
        param_dict['generator'] = [(d['model'], d['config']) for d in\
            param_dict['generator']]
    param_names = param_dict.keys()
    param_vals = [param_dict[k] for k in param_names]

    # Number of expected models & subsampling prob. range
    n_total_configs   = len(list(product(*param_vals)))
    n_expected_models = args.n_models if args.n_models > 0 else n_total_configs
    param_keep_prob   = min(float(n_expected_models) / n_total_configs, 1.0)

    # Fix datetime here so all runs get same one
    log_path = get_log_dir_path(args.log_root, name)
    print("Log path:", log_path)
    
    # Iterate over the param configs and launch subprocesses
    run_idxs = []
    procs    = []
    j        = -1
    for param_set in product(*param_vals):
        if np.random.rand() > param_keep_prob:
            continue
        j += 1
        run_idxs.append(j)

        # Assemble command line argument
        proc_args = [
            'python', args.script, 
            '--run_name', name,
            '--run_index', str(j),
            '--log_path', log_path
        ]
        for k, v in zip(param_names, param_set):
            if k == 'generator':
                proc_args += ['--generator', v[0]]
                proc_args += ['--gen_config', create_config_str(v[1])]
            else:
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
