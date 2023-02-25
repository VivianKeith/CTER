"""
Some simple logging functionality,customized from SpinningUp's logx.py

(https://github.com/openai/spinningup/blob/master/spinup/utils/logx.py)


Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""

import imp
import json
from pickle import FALSE
from pydoc import describe
from tkinter.tix import Tree
from turtle import color
import joblib
import shutil
import numpy as np
import tensorflow as tf
import torch
import os.path as osp, time, atexit, os
import warnings

from .serialization import convert_json
from .config import (
    color2num,
    FORCE_DATESTAMP,
    DEFAULT_DATA_DIR,
    DEFAULT_SEED
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics for.

        with_min_and_max(bool): If true, return min and max of x in addition to mean and std.

    Returns:
        statistics(dict): statistics of x
    """
    statistics = {}
    x = np.array(x, dtype=np.float32)
    statistics["mean"] = np.mean(x)
    statistics["std"] = np.std(x)
    if with_min_and_max:
        statistics["min"] = np.min(x)
        statistics["max"] = np.max(x)
    return statistics

def setup_logger_output_dir(self, exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.
    If no seed is given and datestamp is false, 
    ::

        output_dir = data_dir/exp_name

    If a seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name/exp_name_s[seed]

    If datestamp is true, amend to
    ::
        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in ``config.py``. 

    Args:
        exp_name (string): Name for experiment.

        seed (int): Seed for random number generators used by experiment.

        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``config.py``.

        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.

    Returns:
        output_dir (string): A directory for saving results to.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])
    
    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)

    data_dir = data_dir or DEFAULT_DATA_DIR
    output_dir = output_dir=osp.join(data_dir, relpath)

    return output_dir


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the 
    state of a training run, and the trained model.
    """

    def __init__(self, output_fname='progress.txt', exp_name=None, seed=None, data_dir=None, datestamp=False, **kwargs):
        """
        Initialize a Logger.

        Args:
            output_fname (string): Name for the tab-separated-value file 
                containing metrics logged throughout a training run. 
                Defaults to ``progress.txt``. 

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
            
            seed (int): Seed for random number generators used by experiment.

            data_dir (string): Path to folder where results should be saved.
                Default is the ``DEFAULT_DATA_DIR`` in ``config.py``.

            datestamp (bool): Whether to include a date and timestamp in the
                name of the save directory.

        """
        assert exp_name is not None, "You have to set the exp_name for logger."
        self.output_dir = self.setup_logger_output_dir(exp_name, seed, data_dir, datestamp)
        print("output dir:", self.output_dir)
        if osp.exists(self.output_dir):
            print("Warning: Log dir %s already exists! Storing info there anyway."%self.output_dir)
        else:
            os.makedirs(self.output_dir)
        self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print(colorize("Logging data to %s"%self.output_file.name, 'green', bold=True))

        self.first_row=True
        self.log_headers = []   # headers of tabular diagnostics
        self.log_current_row = {}
        self.exp_name = exp_name

    def setup_logger_output_dir(self, exp_name, seed=None, data_dir=None, datestamp=False):
        """
        Sets up the output_dir for a logger.
        If no seed is given and datestamp is false, 
        ::

            output_dir = data_dir/exp_name

        If a seed is given and datestamp is false,
        ::
            output_dir = data_dir/exp_name/exp_name_s[seed]

        If datestamp is true, amend to
        ::
            output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

        You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in ``config.py``. 

        Args:
            exp_name (string): Name for experiment.

            seed (int): Seed for random number generators used by experiment.

            data_dir (string): Path to folder where results should be saved.
                Default is the ``DEFAULT_DATA_DIR`` in ``config.py``.

            datestamp (bool): Whether to include a date and timestamp in the
                name of the save directory.

        Returns:
            output_dir (string): A directory for saving results to.
        """

        # Datestamp forcing
        datestamp = datestamp or FORCE_DATESTAMP

        # Make base path
        ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
        relpath = ''.join([ymd_time, exp_name])
        
        if seed is not None:
            # Make a seed-specific subfolder in the experiment directory.
            if datestamp:
                hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
            else:
                subfolder = ''.join([exp_name, '_s', str(seed)])
            relpath = osp.join(relpath, subfolder)

        data_dir = data_dir or DEFAULT_DATA_DIR
        output_dir = output_dir=osp.join(data_dir, relpath)

        return output_dir

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""

        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """

        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible). 

        Example use:
            >>> logger = Logger(**logger_kwargs)
            >>> logger.save_config(locals())
        """
        config_json = convert_json(config)
        if config_json.get('exp_name') is None and self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(self.output_dir, "expconfig.json"), 'w') as out:
            out.write(output)

    def save_state(self, state_dict, itr=None):
        """
        Saves the state of an experiment.

        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you 
        previously set up saving for with ``setup_tf_saver``. 

        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent 
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.

        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.

            itr: An int, or None. Current iteration of training.
        """

        fname = 'vars.pkl' if itr is None else 'vars%d.pkl'%itr
        try:
            joblib.dump(state_dict, osp.join(self.output_dir, fname))
        except:
            self.log('Warning: could not pickle state_dict.', color='red')
        if hasattr(self, 'tf_saver_elements'):
            self._tf_simple_save(itr)
        if hasattr(self, 'pytorch_saver_elements'):
            self._pytorch_simple_save(itr)

    def setup_tf_saver(self, sess, inputs, outputs):
        """
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the 
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {'inputs': {k:v.name for k,v in inputs.items()},
                              'outputs': {k:v.name for k,v in outputs.items()}}

    def _tf_simple_save(self, itr=None):
        """
        Uses simple_save to save a trained model, plus info to make it easy
        to associated tensors to variables after restore. 
        """

        assert hasattr(self, 'tf_saver_elements'), \
            "First have to setup saving with self.setup_tf_saver"
        fpath = 'tf1_save' + ('%d'%itr if itr is not None else '')
        fpath = osp.join(self.output_dir, fpath)
        if osp.exists(fpath):
            # simple_save refuses to be useful if fpath already exists,
            # so just delete fpath if it's there.
            shutil.rmtree(fpath)
        tf.saved_model.simple_save(export_dir=fpath, **self.tf_saver_elements)
        joblib.dump(self.tf_saver_info, osp.join(fpath, 'model_info.pkl'))
    
    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.

        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to 
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.

        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, itr=None):
        """
        Saves the PyTorch model (or models).
        """

        assert hasattr(self, 'pytorch_saver_elements'), \
            "First have to setup saving with self.setup_pytorch_saver"
        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, fpath)
        fname = 'model' + ('%d'%itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # I am using a non-recommended way of saving PyTorch models, by pickling whole objects
            # (which are dependent on the exact directory structure at the time of saving) as opposed to just saving network weights. 
            # This works sufficiently well for the general purposes of me, but you may want to do 
            # something different for your personal PyTorch project (e.g. save/load state_dict).
            # We use a catch_warnings() context to avoid the warnings about not being able to save the source code.
            # More info: https://pytorch.org/tutorials/beginner/saving_loading_models.html

            torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the tabular diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """

        vals = []

        # set tabular format
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = '%' + '%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len

        # write to stdout
        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)
        print("-"*n_slashes, flush=True)
        
        # write to the output file
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers)+"\n")
            self.output_file.write("\t".join(map(str,vals))+"\n")
            self.output_file.flush()
        
        # for next row writing
        self.log_current_row.clear()
        self.first_row=False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to 
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use:

    >>> epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you 
    would use:

    >>> epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical 
        values.
        """
        for k,v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with 
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the 
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else "Average" + key, stats["mean"])
            if not average_only:
                super().log_tabular('Std'+key, stats["std"])
            if with_min_and_max:
                super().log_tabular('Min'+key, stats["min"])
                super().log_tabular('Max'+key, stats["max"])
        
        # for next epoch saving
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.

        Call this function before calling log_tabular(), because log_tabular will clear self.epoch_dict[key]
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return statistics_scalar(vals)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog="rltookit.log", description="Test functionalities in rltookit.log.")
    parser.add_argument('--simpletest', action='store_true',
                        help="Run a simple EpochLogger test.")
    parser.add_argument('--rltest', action='store_true',
                        help="Run a EpochLogger test with DQN.")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="Seed for random number generators used by experiment, which determines the postfix of 3rd level of save directory.")
    parser.add_argument('--exp_name', type=str, default='dqntest',
                        help='Experiment name, which determines the 2nd level of save directory.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to folder where results should be saved, which determines the toppest level of save directory.')
    parser.add_argument('--datestamp', '-dt', action='store_true', default=False,
                        help='Whether to include a date and timestamp in the name of the save directory.')
    parser.add_argument('--render', action='store_true',
                        help="Whether to render the CartPole env.")

    args = parser.parse_args()

    if args.simpletest:
        # test colorize
        msg = colorize("HELLO: %s" % "My RL logger", color="red", bold=True, highlight=True)
        print(msg)

        # test logger tabular
        epoch_logger = EpochLogger(exp_name="simpletest", seed=1, data_dir="data", datestamp=False)
        print(colorize("test data:", color='yellow'))
        for i in range(10):
            data_i = (i+1) * 10
            print('data%d:'%(i+1), data_i)
            epoch_logger.store(Test=data_i)
        stat = epoch_logger.get_stats('Test')
        print("simple statistic dict of test data:\n", stat)
        print(colorize("Tabular statistic of test data:"%stat["mean"], color="green"))
        epoch_logger.log_tabular('Test', with_min_and_max=True)
        epoch_logger.dump_tabular()

    if args.rltest:
        # test epoch logger during training a dqn agent
        from .algo_demo.dqn import dqn
        configs = dict(
            exp_name=args.exp_name,
            seed=args.seed,
            data_dir=args.data_dir,
            datestamp=args.datestamp
        )
        dqn(render=args.render, **configs)