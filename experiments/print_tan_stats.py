from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os
import pandas as pd

from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compile and print TAN training run data")
    parser.add_argument('--log_root', type=str,
        help="""Path to run logs, e.g.
        'experiments/log/2017_04_12/tan_only_12_28_44'""")
    parser.add_argument('--sort_by', type=str, default='rand_to_gen_loss')
    parser.add_argument('--desc', type=bool, default=True)
    args = parser.parse_args()

    # Get TAN log directory
    log_root = os.path.join(args.log_root, 'tan')

    # Use the keys from the first run, which we assume has run_index=0
    with open(os.path.join(log_root, '0', 'logs', 'run_log.json'), 'rb') as f:
        keys = json.load(f).keys()

    # Get run_logs from each run
    data = defaultdict(list)
    for run_index in os.listdir(log_root):
        fname = os.path.join(log_root, run_index, 'logs', 'run_log.json')
        with open(fname, 'rb') as f:
            log = json.load(f)
        for k in keys:
            if k in log:
                data[k].append(log[k])
            else:
                data[k].append(None)

    # Only keep the params that were actually varied
    for k in keys:
        try:
            if len(set(data[k])) == 1:
                del data[k]
        except TypeError:
            pass
    keys = data.keys()

    # Take the last element of lists
    for k in keys:
        for i in range(len(data[k])):
            if isinstance(data[k][i], (list, tuple)):
                data[k][i] = data[k][i][-1]

    ###
    ### Add custom attribs here
    ###
    index = data['run_index']

    # rand_to_gen_loss
    rand_to_gen_loss = []
    for i in range(len(index)):
        g_loss = data['G Loss'][i]
        r_loss = data['R Loss'][i]
        both_loss = r_loss is not None and g_loss is not None
        x = float(r_loss) / float(g_loss) if both_loss else 0.0
        rand_to_gen_loss.append(x)
    data['rand_to_gen_loss'] = rand_to_gen_loss

    # Form dataframe
    del data['run_index']
    pd.set_option('display.width', 150)
    print(pd.DataFrame(data=data, index=index).sort_values(
        by=args.sort_by, ascending=False))
