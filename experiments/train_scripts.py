import numpy as np
import os
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict
from experiments.utils import *
from functools import partial
from tanda.discriminator import DCNN, ResNetDefault
from tanda.generator import (
    GRUGenerator, LSTMGenerator, MeanFieldGenerator
)
from tanda.tan import TAN
from tanda.transformer import ImageTransformer, PadCropTransformer
from utils import parse_config_str


#####################################################################
GENERATORS = {
    'gru':        GRUGenerator,
    'lstm':       LSTMGenerator,
    'mean_field': MeanFieldGenerator,
}

TRANSFORMERS = {
    'image':     ImageTransformer,
    'image-pct': PadCropTransformer
}

DISCRIMINATORS = {
    'dcnn':      DCNN,
    'resnet':    ResNetDefault,
}

OPTIMIZERS = {
    'momentum': partial(tf.train.MomentumOptimizer, momentum=0.9),
    'sgd':      tf.train.GradientDescentOptimizer,
    'adam':     tf.train.AdamOptimizer,
}

LR_SCHEDULES = [
    [(0, 0.1), (8000, 0.01), (12000, 0.001)],
]
#####################################################################


#####################################################################
# TF input flags for all basic configurable options
# These should be for run-time configurable options common across all setups
# Dataset / model-specific options are input as args to run
flags = tf.flags

# Input data
flags.DEFINE_integer("subsample_seed", 1701,
    "Random seed to set before data subsampling")

# Run ids
flags.DEFINE_string("run_name", None, "Name of run")
flags.DEFINE_integer("run_index", 0, "Index in run")

# Run type--see `train()` for descriptions
flags.DEFINE_string("run_type", "tanda-full", "Type of run")

# TAN params
flags.DEFINE_float("gamma", 0.5, "Discount future for future rewards")
flags.DEFINE_boolean("per_img_std", True, "Apply per-image standardization")
flags.DEFINE_float("mse_term", 1e-4,
    "Coefficient for MSE term in gen model objective")
flags.DEFINE_integer("mse_layer", -1,
    ("Number of disc model layers back from linear layer to compute MSE. " +
    "If None or < 0, pixel distance."))
flags.DEFINE_integer("batch_size", 100, "number of samples per batch")
flags.DEFINE_integer("n_epochs", 5, "Number of training epochs")
flags.DEFINE_string("tan_checkpoint_path", None,
    "Checkpoint path for trained TAN")

# Transformer
flags.DEFINE_string("transformer", "image", "Transformer class")

# Generator params
flags.DEFINE_string("generator", "gru", "Generator class to use")
flags.DEFINE_string("gen_config", "n_stack=1,feed_actions=True,init_type=zeros",
    "String describing generator config; see utils.create_config_str")
flags.DEFINE_integer("seq_len", 15, "TF sequence length")
flags.DEFINE_float("gen_lr", 1e-3, "generator learning rate")
flags.DEFINE_integer("n_gen_steps", 1, "Number of steps to take w gen model")
flags.DEFINE_integer("n_sample", 10, "Number of sequences to sample per point")
flags.DEFINE_integer("n_tan_train", 0, 
    "Number of training data pts. to use when training TAN")

# Discriminator params
flags.DEFINE_boolean("train_disc", True, "Whether to train discriminator")
flags.DEFINE_string("discriminator", "dcnn", "Discriminator class to use")
flags.DEFINE_float("disc_lr", 1e-5, "discriminator learning rate")
flags.DEFINE_integer("n_disc_steps", 1, "Number of steps to take w disc. model")

# End model params
flags.DEFINE_string("end_discriminator", "resnet", "End discriminator class")
flags.DEFINE_integer("end_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("end_batch_size", 50, "Batch size to use for end model")
flags.DEFINE_boolean("end_per_img_std", True, "Apply per-image standardization")
flags.DEFINE_integer("n_per_class", 500,
    "Number of data pts. per class for end model")
# End p_transform
flags.DEFINE_float("p_transform_init", 0.1,
    "Prob. of transforming a data pt. in end model epoch")
flags.DEFINE_float("p_transform_rate", 1.05, "Rate of change of p_transform")
flags.DEFINE_float("p_transform_max", 0.5,
    "Prob. of transforming a data pt. in end model epoch")
flags.DEFINE_float("p_transform_min", 0.5,
    "Prob. of transforming a data pt. in end model epoch")
flags.DEFINE_integer("p_transform_drop", None,
    "Prob. of transforming a data pt. in end model epoch")
# End LR
flags.DEFINE_string("end_lr_mode", "constant", "{constant,schedule}")
flags.DEFINE_float("end_lr", 1e-3, "End model LR if end_lr_mode=constant")
flags.DEFINE_integer("end_lr_schedule", 0, 
    "Index of end lr schedule if end_lr_mode=schedule")
# Regularization parameters
flags.DEFINE_float("end_weight_decay", 0.0002, "Weight decay for end model")
# End optimizer
flags.DEFINE_string("end_optimizer", "momentum", "End optimizer")
# Saving checkpoints
flags.DEFINE_integer("save_end_model_every", 50,
    "Epoch frequency at which to save end model checkpoints")
# Whether to use unlabeled data local smoothness reg. term
flags.DEFINE_float("ls_term", 0.1, "Unlabeled data local smoothness term")
flags.DEFINE_integer("ls_term_n_passes", 1, "N passes per data point")
flags.DEFINE_integer("end_batch_size_u", 10,
    "Batch size for unlabeled data for LS reg term")

# Training fold params
flags.DEFINE_integer("n_folds", -1,
    "Number of folds to average over")
flags.DEFINE_integer("run_fold", 0, "Fold to run")

# Output params
flags.DEFINE_string("log_path", None, "Log path")
flags.DEFINE_boolean("is_test", False, "Is test?")
flags.DEFINE_boolean("save_model", True, "Save the model after every epoch?")
flags.DEFINE_string("log_root", "experiments/log", "Root directory for logging")
flags.DEFINE_integer("eval_every", 1, "Period at which to eval accuracy")
flags.DEFINE_integer("rand_loss_every", 1,
    "Period at which to compute random batch loss")
flags.DEFINE_integer("plot_every", 50, "Period at which to plot images")
flags.DEFINE_string("date_stamp", None, "Date stamp for logs")
flags.DEFINE_string("time_stamp", None, "Time stamp for logs")
flags.DEFINE_boolean("save_action_seqs", True, "Write out action seqs to file.")

# For testing transforming the test set as well
flags.DEFINE_boolean("transform_validation_set", False,
    "Transform validation set as well")

FLAGS = flags.FLAGS

#####################################################################
# Top-level method to be used
#####################################################################

def train(X_train, dims, tfs, Y_train=None, X_valid=None, Y_valid=None,
    n_classes=None, **kwargs):
    """Highest-level generic training script.
    High-level route is decided by FLAGS.run_type:
        - 'tan-only'  :       Only train the TAN
        - 'tanda-full':       Train TAN -> use it for DA with end model
        - 'tanda-pretrained': Only run a pre-trained TAN
        - 'random':           Run end model with random DA
        - 'baseline':         Run end model with no DA
        - 'basic':            Pass through basic transform of transformer
    """
    # Create log path: Create at this level (or one above, in launch script)
    # so that we use same one for both steps
    log_path = FLAGS.log_path if FLAGS.log_path is not None else \
        get_log_dir_path(FLAGS.log_root, FLAGS.run_name)
    
    ###
    ### STEP 1: TRAIN TAN
    ###
    # If a TAN-only or full TANDA run, train
    if FLAGS.run_type in ['tanda-full', 'tan-only']:
        if FLAGS.is_test:
            print "STEP 1: Training TAN"
        # Get trained TAN + its checkpoint directory since graph will be reset
        tan, tan_checkpoint_path = train_tan(
            X_train, dims, tfs, log_path, **kwargs
        )
    # If tanda-pretrained, load TAN from path provided in FLAGS
    elif FLAGS.run_type == 'tanda-pretrained':
        tan                 = assemble_tan(dims, tfs, **kwargs)
        tan_checkpoint_path = FLAGS.tan_checkpoint_path
    elif FLAGS.run_type in ['random', 'baseline', 'basic']:
        tan                 = assemble_tan(dims, tfs, **kwargs)
        tan_checkpoint_path = None
    else:
        raise ValueError("Run type %s not recognized" % FLAGS.run_type)

    ###
    ### STEP 2: TRAIN END MODEL
    ###

    if FLAGS.run_type != 'tan-only':
        if FLAGS.is_test:
            print "STEP 2: Training End Model"
        train_end_model(X_train, Y_train, X_valid, Y_valid, n_classes, dims,
            log_path, tan=tan, tan_checkpoint_path=tan_checkpoint_path)


#####################################################################
# UTIL FUNCTIONS
#####################################################################

def select_fold(X, Y):
    n_per_class = FLAGS.n_per_class * FLAGS.n_folds
    print "N-per-class:", n_per_class
    
    # Set the seed
    assert(FLAGS.subsample_seed > 0)
    np.random.seed(FLAGS.subsample_seed)
    
    # Construct the folds
    class_idxs = get_class_idxs(Y)
    
    # Make sure all classes have correct number of examples
    for c in class_idxs:
        idxs = class_idxs[c]
        np.random.shuffle(idxs)
        k = len(idxs)
        if k > n_per_class:
            class_idxs[c] = idxs[:n_per_class]
        elif k < n_per_class:
            class_idxs[c] = idxs + idxs[:n_per_class-k]
    
    # Check that all classes have same number of instances and this
    # number is divides evenly into folds
    print ' '.join(str(len(v)) for v in class_idxs.values())
    assert(all(len(v) == n_per_class for v in class_idxs.values()))
    assert(n_per_class % FLAGS.n_folds == 0)
    n_per_fold = n_per_class / FLAGS.n_folds
    
    # Divide into folds
    folds = [[] for _ in range(FLAGS.n_folds)]
    for idxs in class_idxs.values():
        np.random.shuffle(idxs)
        for i, k in enumerate(range(0, n_per_class, n_per_fold)):
            folds[i].extend(idxs[k : k + n_per_fold])
    
    # Get the fold we want
    fold_samp = sorted(folds[FLAGS.run_fold])[:10]
    print "CONSTRUCTED FOLDS WITH {0} EXAMPLES PER CLASS PER FOLD".format(
        n_per_fold
    )
    print "RUNNING FOLD {0}: [{1}]...".format(
        FLAGS.run_fold, ' '.join(map(str, fold_samp))
    )
    FOLD = np.ravel(folds[FLAGS.run_fold])
    np.random.shuffle(FOLD)
    return X[FOLD], Y[FOLD]


def assemble_tan(dims, tfs, d_class=None, t_class=None, t_kwargs={}):
    """Assembles TAN based on FLAGS"""

    # Cmd-line arg selection of generator
    if FLAGS.generator in GENERATORS:
        gen_class = GENERATORS[FLAGS.generator]
    else:
        raise ValueError("Unrecognized generator class: %s" % FLAGS.generator)

    # Cmd-line arg selection OR custom class of discriminator
    if d_class is None and FLAGS.discriminator in DISCRIMINATORS:
        d_class = DISCRIMINATORS[FLAGS.discriminator]
    elif d_class is None:
        raise ValueError("Unrecognized disc. class: %s" % FLAGS.discriminator)

    # Transformer: For now, use ImageTransformer by default
    if t_class is not None:
        transformer = t_class(tfs, **t_kwargs)
    elif FLAGS.transformer in TRANSFORMERS:
        transformer = TRANSFORMERS[FLAGS.transformer](
            tfs, dims=dims, **t_kwargs)
    else:
        raise ValueError("Unrecognized trans. class: %s" % FLAGS.transformer)

    K = transformer.n_actions  # NOTE: This is *not* necessarily len(tfs)

    # Build generator
    G = gen_class(K, FLAGS.seq_len, **parse_config_str(FLAGS.gen_config))
    # Build discriminator
    D = d_class(dims=dims)
    # MSE layer
    mse_layer = FLAGS.mse_layer if FLAGS.mse_layer > 0 else None
    # Build TAN
    return TAN(D, G, transformer, FLAGS.disc_lr, FLAGS.gen_lr,
        mse_term=FLAGS.mse_term, mse_layer=mse_layer,
        gamma=FLAGS.gamma, per_img_std=FLAGS.per_img_std,
        train_disc=FLAGS.train_disc)


#####################################################################
# TRAIN_TAN FUNCTION
#####################################################################


def train_tan(X, dims, tfs, log_path, d_class=None, t_class=None,
    t_kwargs={}, plotter=None):

    # Shuffle and optionally subsample data
    N_train = FLAGS.n_tan_train if FLAGS.n_tan_train > 0 else X.shape[0]
    idxs = range(X.shape[0])
    if FLAGS.subsample_seed > 0:
        np.random.seed(FLAGS.subsample_seed)
    np.random.shuffle(idxs)
    idxs = idxs[:N_train]
    X = X[idxs]

    # Reshape / flatten data
    # Note: Should we be flattening?
    dims_flat = np.prod(dims)
    if [dims_flat] != dims:
        X = X.reshape([N_train, dims_flat])

    # Create time stamped directories for logs, plots, and checkpoints
    LOGDIR, PLOTDIR, SAVEDIR = create_subdirs(log_path, 'tan', FLAGS.run_index)
    if FLAGS.is_test:
        print "LOGDIR: %s" % LOGDIR

    # Assemble TAN model based on FLAGS
    tan = assemble_tan(
        dims, tfs, d_class=d_class, t_class=t_class, t_kwargs=t_kwargs
    )
    if FLAGS.is_test:
        # Print TAN generator and discriminator var counts
        tan_vars = tf.trainable_variables()
        tan_vars_g = filter(lambda v : v.name.startswith('gen'), tan_vars)
        nvg, _ = slim.model_analyzer.analyze_vars(tan_vars_g, print_info=False)
        tan_vars_d = filter(lambda v : v.name.startswith('disc'), tan_vars)
        nvd, _ = slim.model_analyzer.analyze_vars(tan_vars_d, print_info=False)
        tan_vars_o = filter(
            lambda v : re.search(r'^gen|disc', v.name) is None, tan_vars)
        nvo, _ = slim.model_analyzer.analyze_vars(tan_vars_o, print_info=False)
        print "# vars: {0} gen, {1} disc, {2} other".format(nvg, nvd, nvo)

    # Initialize and save log file
    log_dict = create_run_log(LOGDIR, FLAGS)

    # As default create ImagePlotter for routing images into Tensorboard
    if plotter is None and FLAGS.plot_every > 0:
        img_plotter = ImagePlotter(dims, FLAGS.batch_size, PLOTDIR,
            ['plot_baseline', 'plot_random', 'plot_tanda'])

    # Add diversity statistic plotters
    if FLAGS.save_action_seqs:
        jaccard_plotter  = ConstantPlotter('avg_jaccard', tf.float32)
        ng_ratio_plotter = ConstantPlotter('ngram_ratio', tf.float32)
    
    # Train TAN
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # Intantiate FileWriter for Tensorboard
        writer = tf.summary.FileWriter(LOGDIR, session.graph)

        # Run minibatch training steps
        n_batches = int(np.ceil(N_train / float(FLAGS.batch_size)))
        for t in xrange(FLAGS.n_epochs):

            # Iterate over batches
            d_losses, g_losses, r_losses = [], [], []
            for i, b in enumerate(range(0, N_train, FLAGS.batch_size)):

                # Get next batch
                X_batch = X[b : b + FLAGS.batch_size]
                # Training step
                d_loss, g_loss, summary, deltas, tf_seqs = tan.train_step(
                        session, X_batch, FLAGS.n_disc_steps, FLAGS.n_gen_steps,
                        n_sample=FLAGS.n_sample
                    )
                d_losses.append(d_loss)
                g_losses.append(g_loss)

                # Write tensorboard events
                c = t * n_batches + i
                writer.add_summary(summary, c)

                # Write out action sequences and plot
                if FLAGS.save_action_seqs:
                    with open('%s/actions_epoch_%s' % (LOGDIR, t), 'a+') as f:
                        for ii in range(tf_seqs.shape[0]):
                            f.write(",".join(map(str, tf_seqs[ii])) + "\n")
                    # Compute summary statistics
                    j_dist = average_all_pairs_jaccard_distance(tf_seqs)
                    writer.add_summary(
                        jaccard_plotter.get_summary(session, j_dist), c)
                    ng_ratio = ngram_ratio(tf_seqs, tan.generator.m, n=3)
                    writer.add_summary(
                        ng_ratio_plotter.get_summary(session, ng_ratio), c)

                # Compute random loss too periodically
                # Note that this takes non-negligible time, hence periodic eval
                if FLAGS.rand_loss_every > 0 and i % FLAGS.rand_loss_every == 0:
                    r_loss, r_summary, gr_summary = tan.get_random_loss(
                        session, X_batch, gen_loss=g_loss)
                    r_losses.append(r_loss)
                    writer.add_summary(r_summary, c)
                    writer.add_summary(gr_summary, c)

                # Print losses
                if FLAGS.is_test:
                    info = [
                        ('Epoch', str(t)),
                        ('Batch', "{0} / {1}".format(i+1, n_batches)),
                        ('D', np.mean(d_losses)), ('G', np.mean(g_losses)),
                        ('R', np.mean(r_losses))
                    ]
                    line_writer(OrderedDict(info))
            
                # Plot transformed images in batch
                if FLAGS.plot_every > 0 and i % FLAGS.plot_every == 0:
                    # Default plotter
                    if plotter is None:
                        X_plot   = X[:FLAGS.batch_size]
                        X_plot_b = tan.transformer.transform_basic(
                            X_plot, train=False)
                        X_plot_t = tan.get_transformed_data(session, X_plot)[0]
                        X_plot_r = tan.transformer.random_transform(
                            X_plot, FLAGS.seq_len, emit_incremental=False
                        )
                        Xs = [X_plot_b, X_plot_r, X_plot_t]
                        plot_summary = img_plotter.get_image_summaries(
                            session, Xs, 'img_{0}_{1}'.format(t, i))
                        writer.add_summary(plot_summary, c)
                    # This is pretty hacky... is just for the synthetic data
                    # experiment
                    else:
                        plotter(session, tan, X, FLAGS.batch_size,
                            '{0}/img_{1}_{2}'.format(PLOTDIR, t, i))
                    
            # Print average batch losses
            if FLAGS.is_test:
                info = OrderedDict([
                    ('Epoch', str(t)),
                    ('Batch', "{0} / {1}".format(i+1, n_batches)),
                    ('D [avg]', np.mean(d_losses)),
                    ('G [avg]', np.mean(g_losses)),
                    ('R [avg]', np.mean(r_losses))
                ])
                line_writer(info)

            # Save average losses to run log
            log_dict['D Loss'].append(float(np.mean(d_losses)))
            log_dict['G Loss'].append(float(np.mean(g_losses)))
            log_dict['R Loss'].append(float(np.mean(r_losses)))
            save_run_log(log_dict, LOGDIR)

            # After each epoch save the model
            if FLAGS.save_model:
                tan.save(session, os.path.join(SAVEDIR, 'tan_checkpoint'))
                if FLAGS.is_test:
                    print "Saved model"

    # Return trained TAN + checkpoint save dir
    return tan, SAVEDIR


#####################################################################
# TRAIN_END_MODEL FUNCTION
#####################################################################


def transform_batch(X_batch, tan, sess, run_type, p_transform=1.0):
    """Transforms a portion of a batch of data according to a certain mode"""
    # Pass _all_ data through transform basic (can skip if p_transform = 1.0)
    if p_transform < 1.0:
        X_batch_t = tan.transformer.transform_basic(
            np.copy(X_batch), train=False)
    else:
        X_batch_t = np.copy(X_batch)

    if p_transform > 0.0:
        # Select the data points in the batch to transform
        t_idxs = range(min(FLAGS.end_batch_size, X_batch.shape[0]))
        np.random.shuffle(t_idxs)
        t_idxs = t_idxs[:int(FLAGS.end_batch_size * p_transform)]
        
        # Transform according to run_type
        if run_type == 'tanda':
            X_batch_t[t_idxs], _, _ = tan.get_transformed_data(
                sess,
                X_batch[t_idxs, :],
                emit_incremental=False,
                n_seqs_per_example=1
            )
        elif run_type == 'random':
            X_batch_t[t_idxs] = tan.transformer.random_transform(
                X_batch[t_idxs, :],
                FLAGS.seq_len,
                emit_incremental=False
            )
        elif run_type == 'basic':
            # We apply transform_basic in *train mode* to everything
            X_batch_t = tan.transformer.transform_basic(
                X_batch, train=True)

    return X_batch_t


def train_end_model(X_train_full, Y_train_full, X_valid, Y_valid,
    n_classes, dims, log_path, tan=None, tan_checkpoint_path=None):
    """End model training routine
    Runs a _single_ end model either with no, random, or TAN augmentation.

    @{X,Y}_{train_full,valid}: The training set / validation set with labels
    @n_classes: Number of classes
    @dims: Vector of dimensions for a single data point;
        e.g. for images: [height, width, n_channels]
    @log_path: Path to base log directory for end run
    @tan: If None, no augmentations will be used
    @tan_checkpoint_path: If None but `tan` is not None, random TF sequences
        will be used; else, if both `tan` and `tan_checkpoint_path` are not
        None, will use trained TF sequence generator.
    """
    # Determine which mode we are running in
    if FLAGS.run_type == 'baseline':
        run_type = 'baseline'
    elif FLAGS.run_type == 'random':
        run_type = 'random'
    elif FLAGS.run_type == 'basic':
        run_type = 'basic'
    # This covers tanda-full and tanda-pretrained
    else:
        run_type = 'tanda'

    # Get balanced subsample of data
    if FLAGS.subsample_seed > 0:
        np.random.seed(FLAGS.subsample_seed)
    if FLAGS.n_per_class > 0:
        X_train, Y_train = balanced_subsample(X_train_full, Y_train_full,
            FLAGS.n_per_class)
    else:
        X_train = X_train_full
        Y_train = Y_train_full
    N_train = X_train.shape[0]

    # Reshape / flatten data
    dims_flat = np.prod(dims)
    X_train = X_train.reshape([N_train, dims_flat])
    X_valid = X_valid.reshape([X_valid.shape[0], dims_flat])
    N_u = 0
    if FLAGS.ls_term > 0.0:
        N_u = X_train_full.shape[0]
        X_u = np.copy(X_train_full).reshape([N_u, dims_flat])
    if FLAGS.is_test:
        print "N_train=%s, N_u=%s" % (N_train, N_u)

    # Create time stamped directories for logs, plots, and checkpoints
    LOGDIR, PLOTDIR, SAVEDIR = create_subdirs(
        log_path, 'end_model', FLAGS.run_index
    )
    if FLAGS.is_test:
        print "LOGDIR: %s" % LOGDIR

    # Initialize and save log file
    log_dict = create_run_log(LOGDIR, FLAGS)

    # Create ImagePlotter for routing images into Tensorboard
    plot_names = ['plot_%s' % run_type]
    img_plotter = ImagePlotter(dims, FLAGS.end_batch_size, PLOTDIR, plot_names)

    # Create a ConstantPlotter for plotting p_transform
    p_transform_plotter = ConstantPlotter('p_transform', tf.float32)

    # Cmd-line arg selection of discriminator
    if FLAGS.end_discriminator in DISCRIMINATORS:
        d_class = DISCRIMINATORS[FLAGS.end_discriminator]
    else:
        raise ValueError(
            "Unrecognized disc class argument: %s" % FLAGS.discriminator)
    D = d_class(dims=dims)

    # Learning rate mode
    if FLAGS.end_lr_mode == 'constant':
        lr = FLAGS.end_lr
    elif FLAGS.end_lr_mode == 'schedule':
        lr_schedule = LR_SCHEDULES[FLAGS.end_lr_schedule]
        lr_si = 0
        lr    = lr_schedule[lr_si][1]
    else:
        raise ValueError("end_lr_mode %s not recognized." % FLAGS.end_lr_mode)

    # Build supervised discriminative model
    D.build_supervised(
        n_classes, 
        'D',
        lr_init=lr,
        per_img_std=FLAGS.end_per_img_std,
        trainer=OPTIMIZERS[FLAGS.end_optimizer],
        weight_decay=FLAGS.end_weight_decay,
        ls_term=FLAGS.ls_term,
        ls_term_n_passes=FLAGS.ls_term_n_passes
    )

    # Print end discriminator variables
    if FLAGS.is_test:
        vars = tf.trainable_variables()
        D_vars = filter(lambda v : v.name.startswith('D'), vars)
        nv, _ = slim.model_analyzer.analyze_vars(D_vars, print_info=False)
        print "# end discriminator vars: {0}".format(nv)
    
    # Train end model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Intantiate FileWriters for Tensorboard
        graph  = sess.graph
        writer = tf.summary.FileWriter(LOGDIR, graph)

        # Load trained TAN model if applicable
        if run_type == 'tanda':
            checkpoint = os.path.join(tan_checkpoint_path, 'tan_checkpoint')
            tan.restore(sess, checkpoint)

        # Taper in the transformation rate
        p_transform = FLAGS.p_transform_init if run_type != 'baseline' else 0.0

        # Run minibatch training steps
        n_batches = int(np.floor(N_train / float(FLAGS.end_batch_size)))
        for t in xrange(FLAGS.end_epochs):

            # Iterate over batches
            losses = []
            for i, b in enumerate(range(0, N_train, FLAGS.end_batch_size)):
                # Global iteration counter
                c = t * n_batches + i

                # Get next batch of labeled data
                X_batch = X_train[b : b + FLAGS.end_batch_size, :]
                Y_batch = Y_train[b : b + FLAGS.end_batch_size, :]

                # Transform some portion of the batch
                X_batch_t = transform_batch(
                    X_batch, tan, sess, run_type, p_transform)

                # Optionally pass both labeled + unlabeled data through several
                # transformations, to be used in a transformation regularization
                # (TR) term
                U_batch_ts = None
                if FLAGS.ls_term > 0.0:
                    U_batch_ts = []

                    # Select a random batch of unlabeled data
                    uidxs = range(N_u)
                    np.random.shuffle(uidxs)
                    U_batch = X_u[uidxs[:FLAGS.end_batch_size_u]]

                    # Concatenate with the labeled data
                    U_batch = np.vstack([U_batch, X_batch])

                    # Add the basic-transformed version
                    U_batch_ts.append(
                        tan.transformer.transform_basic(U_batch, train=True)) 

                    # Create multiple transformed copies of the same data points
                    for _ in range(FLAGS.ls_term_n_passes):
                        U_batch_t = transform_batch(
                            U_batch, tan, sess, run_type)
                        U_batch_ts.append(U_batch_t)
                    U_batch_ts = np.array(U_batch_ts)

                # Training step
                loss, summ = D.supervised_train_step(
                    sess, X_batch_t, Y_batch, U_ts=U_batch_ts, lr=lr)
                losses.append(loss)

                # Write tensorboard events
                writer.add_summary(summ, c)

                # Print losses
                if FLAGS.is_test:
                    info = OrderedDict([
                        ('Epoch', str(t)),
                        ('Batch', "%s / %s" % (i, n_batches)),
                        ('Loss', loss)
                    ])
                    line_writer(info, newline=True)

                # Plot transformed images in batch
                if FLAGS.plot_every > 0 and i % FLAGS.plot_every == 0:
                    p = img_plotter.get_image_summaries(sess, [X_batch_t],
                        'img_{0}_{1}'.format(t, i))
                    writer.add_summary(p, c)

            # Save model checkpoint
            if (t + 1) % FLAGS.save_end_model_every == 0:
                D.save(sess, os.path.join(SAVEDIR, 'model_checkpoint-%s' % t))

            # Option to save time by evaluating accuracy less often
            if (t + 1) % FLAGS.eval_every == 0:

                # Optionally transform the evaluation set as well
                ptv = p_transform if FLAGS.transform_validation_set else 0.0
                X_valid_t = transform_batch(X_valid, tan, sess, run_type, ptv)

                # Evaluate the model on the validation set
                acc, acc_summ = D.get_accuracy(sess, X_valid_t, Y_valid,
                    batch_size=FLAGS.end_batch_size)
                writer.add_summary(acc_summ, c)

                if FLAGS.is_test:
                    info = OrderedDict([
                        ('Epoch', str(t)),
                        ('Acc', acc),
                        ('Mean Batch Loss', np.mean(losses)),
                    ])
                    line_writer(info, newline=True)

                # Save log file
                log_dict['Loss'].append(float(np.mean(losses)))
                log_dict['Acc'].append(float(acc))
                save_run_log(log_dict, LOGDIR)

            # Increase p_transform
            if run_type != 'baseline':
                if FLAGS.p_transform_drop and t >= FLAGS.p_transform_drop:
                    p_transform = 0.0
                elif FLAGS.p_transform_rate > 1.0:
                    p_transform = min(
                        FLAGS.p_transform_max,
                        p_transform * FLAGS.p_transform_rate
                    )
                else:
                    p_transform = max(
                        FLAGS.p_transform_min,
                        p_transform * FLAGS.p_transform_rate
                    )
            writer.add_summary(
                p_transform_plotter.get_summary(sess, p_transform), c)
            if FLAGS.is_test:
                print 'p_transform=%s' % p_transform

            # Optionally decay learning rate
            if FLAGS.end_lr_mode == 'schedule':
                if lr_si < len(lr_schedule) - 1 and c > lr_schedule[lr_si+1][0]:
                    lr_si += 1
                    lr = lr_schedule[lr_si][1]
                if FLAGS.is_test:
                    print 'lr=%s' % lr

            # Reshuffle
            idxs = range(X_train.shape[0])
            np.random.shuffle(idxs)
            X_train = X_train[idxs]
            Y_train = Y_train[idxs]
