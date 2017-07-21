# NOTE!!! IMPORT utils FIRST SO THAT MATPLOTLIB DOESN'T GET MESSED UP!!!
from utils import (
    generate_data, save_data_plot, OracleDiscriminator
)
from experiments.tfs.image import *
from experiments.train_scripts import (
    flags, train_tan, train_end_model, assemble_tan
)
from experiments.utils import get_log_dir_path
from functools import partial
from tanda.discriminator import SimpleDiscriminator
from tanda.transformer import Transformer


#####################################################################
# Additional TF input flags
flags.DEFINE_integer("tfs", 1, "TF set to use")

# Synthetic datset generation
flags.DEFINE_integer("synthetic_n", 1000,
    "Number of training points to generate")
flags.DEFINE_integer("synthetic_dim", 2, "Dimension of synthetic data")
flags.DEFINE_float("synthetic_r", 1.0,
    "Radius of ball around origin in which data is uniformly generated")

# Using an oracle discriminator
flags.DEFINE_boolean("oracle_disc", False,
    "Optionally use a perfect discriminator, for testing")

FLAGS = flags.FLAGS
#####################################################################


#####################################################################
# Transformation functions
tf_sets = []
d = FLAGS.synthetic_dim

### TF SET 1: Small vs. Large
def TF_displace(x, d=0):
    """Displace point by vector d"""
    return x + d

small_disps = [0.2 * (np.random.random(d) - 0.5) for _ in xrange(10)]
large_disps = [4.0 * (np.random.random(d) - 0.5) for _ in xrange(5)]

if FLAGS.is_test and FLAGS.tfs == 0:
    for disp in small_disps + large_disps:
        print disp

tfs_1 = [partial(TF_displace, d=disp) for disp in small_disps + large_disps]
tf_sets.append(tfs_1)


### TF SET 2: Medium displacements along the main axes
# Note that:
# * Two or more displacements along same axis = bad
# * We will have FLAGS.synthetic_dim number of displacements, each with
#   magnitude 0.75
# * Should set sequence_length to be = FLAGS.synthetic_dim
medium_disps = []
for i in range(d):
    x = np.zeros(d)
    x[i] = 0.75
    medium_disps.append(x)
tfs_2 = [partial(TF_displace, d=disp) for disp in medium_disps]
tf_sets.append(tfs_2)


### TF SET 3: Non-commuting displacements
def TF_displace_decay(x, d=0, r=2.0):
    """Displace point by decay * d, where decay = min(1, r / |x|^2)"""
    return x + min(1.0, r / np.linalg.norm(x)**2) * d

medium_disps_2 = []
for i in range(d):
    x = np.zeros(d)
    #x[i] = np.random.random() + 0.5
    x[i] = 0.75
    medium_disps_2.append(x)
    medium_disps_2.append(-np.copy(x))
tfs_3 = [partial(TF_displace_decay, d=disp) for disp in medium_disps_2]
tf_sets.append(tfs_3)


### TF SET 4: Uniform-magnitude random vectors + non-commuting null zone
def TF_displace_stuck(x, d=0, r=1.5):
    return x + d if np.linalg.norm(x) < r else x


r    = 0.33
vecs = []
for _ in range(10):
    # Pick an angle uniformly
    theta = 2 * np.pi * np.random.random()
    vecs.append(np.array([r * np.cos(theta), r * np.sin(theta)]))
tfs_4 = [partial(TF_displace_stuck, d=v) for v in vecs]
tf_sets.append(tfs_4)


tfs = tf_sets[FLAGS.tfs - 1]

#####################################################################

if __name__ == '__main__':

    # Create log path: Create at this level (or one above, in launch script)
    # so that we use same one for both steps
    log_path = FLAGS.log_path if FLAGS.log_path is not None else \
        get_log_dir_path(FLAGS.log_root, FLAGS.run_name)

    # Note that the flags in this file control the dataset size, not the 
    # normal flags in train_scripts.py!
    dims = [FLAGS.synthetic_dim]
    if FLAGS.subsample_seed > 0:
        np.random.seed(FLAGS.subsample_seed)
    X = generate_data(
        FLAGS.synthetic_n, d=FLAGS.synthetic_dim, r=FLAGS.synthetic_r)

    # For testing, also include a discriminator which is perfectly correct
    if FLAGS.oracle_disc:
        d_class = OracleDiscriminator
    else:
        d_class = SimpleDiscriminator

    ###
    ### STEP 1: TRAIN TAN
    ###
    if FLAGS.is_test:
        print "STEP 1: Training TAN"
    
    train_tan(X, dims, tfs, log_path, d_class=d_class, 
        t_class=Transformer, plotter=save_data_plot)
