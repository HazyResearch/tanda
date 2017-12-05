from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

from experiments.train_scripts import GENERATORS
from experiments.utils import parse_config_str
from keras import backend as K
from tanda.generator import (
    GRUGenerator, LSTMGenerator, MeanFieldGenerator
)
from tanda.tan import PretrainedTAN
from tanda.transformer import PadCropTransformer


def load_pretrained_tan(config_path, checkpoint_path, tfs, dims=[32, 32, 3]):
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Build transformer
    T = PadCropTransformer(tfs, dims=dims)
    # Build generator
    k = T.n_actions
    g_class = GENERATORS[config['generator']]
    G = g_class(k, config['seq_len'], **parse_config_str(config['gen_config']))
    # Build TAN
    return PretrainedTAN(G, T, dims, K.get_session(), checkpoint_path)
