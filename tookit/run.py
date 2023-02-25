"""
Some running functionality supporting CLI, customized from SpinningUp's run.py and run_util.py

(https://github.com/openai/spinningup/blob/master/spinup/run.py)
(https://github.com/openai/spinningup/blob/master/spinup/utils/run_util.py)

Run multiple experiments supporting CLI (Command Line Interface)
"""

import importlib
import sys, os
import time
from textwrap import dedent
import json
import string

import tensorflow as tf
import torch
import numpy as np
from tqdm import trange

from .serialization import convert_json
from .log import colorize
from .config import (
    DEFAULT_DATA_DIR,
    DEFAULT_SHORTHAND,
    FORCE_DATESTAMP,
    WAIT_BEFORE_LAUNCH,
    DEFAULT_SEED,
    RUN_KEYS,
    SUBSTITUTIONS
)

DIV_LINE_WIDTH = 80


def all_bools(vals):
    return all([isinstance(v,bool) for v in vals])


def valid_str(v):
    """ 
    Convert a value or values to a string which could go in a filepath.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    """
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'. 
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


def friendly_err(err_msg):
    # add whitespace to error message to make it more readable
    return '\n\n' + err_msg + '\n\n'


def valid_cmd(cmd):
    try:
        algo_module_path, algo_name = cmd.rsplit('.', 1)
        algo_module = importlib.import_module(algo_module_path)
        algo = getattr(algo_module, algo_name)
        return algo
    except:
        return False


class ExperimentGrid:
    """
    Tool for running many experiments given hyperparameter ranges.
    """

    def __init__(self, name='NaFooExpName'):
        self.keys = []
        self.vals = []
        self.shs = []
        self.in_names = []
        self.name(name)

    def name(self, _name):
        assert isinstance(_name, str), "Name has to be a string."
        self._name = _name

    def print(self):
        """Print a helpful report about the experiment grid."""
        print('='*DIV_LINE_WIDTH)

        # Prepare announcement at top of printing. If the ExperimentGrid has a
        # short name, write this as one line. If the name is long, break the
        # announcement over two lines.
        base_msg = 'ExperimentGrid %s runs over parameters:\n'
        name_insert = '['+self._name+']'
        if len(base_msg%name_insert) <= 80:
            msg = base_msg%name_insert
        else:
            msg = base_msg%(name_insert+'\n')
        print(colorize(msg, color='green', bold=True))

        # List off parameters, shorthands, and possible values.
        for k, v, sh in zip(self.keys, self.vals, self.shs):
            color_k = colorize(k.ljust(40), color='cyan', bold=True)
            print('', color_k, '['+sh+']' if sh is not None else '', '\n')
            for i, val in enumerate(v):
                print('\t' + str(convert_json(val)))
            print()

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_total = int(np.prod([len(v) for v in self.vals]))
        if 'seed' in self.keys:
            num_seeds = len(self.vals[self.keys.index('seed')])
            nvars_seedless = int(nvars_total / num_seeds)
        else:
            nvars_seedless = nvars_total
        print(' Variants, counting seeds: '.ljust(40), nvars_total)
        print(' Variants, not counting seeds: '.ljust(40), nvars_seedless)
        print()
        print('='*DIV_LINE_WIDTH)

    def _default_shorthand(self, key):
        # Create a default shorthand for the key, built from the first 
        # three letters of each colon-separated part.
        # But if the first three letters contains something which isn't
        # alphanumeric, shear that off.
        valid_chars = "%s%s" % (string.ascii_letters, string.digits)
        def shear(x):
            return ''.join(z for z in x[:3] if z in valid_chars)
        sh = '-'.join([shear(x) for x in key.split(':')])
        return sh

    def add(self, key, vals, shorthand=None, in_name=False):
        """
        Add a parameter (key) to the grid config, with potential values (vals).

        By default, if a shorthand isn't given, one is automatically generated
        from the key using the first three letters of each colon-separated
        term. To disable this behavior, change ``DEFAULT_SHORTHAND`` in the
        ``config.py`` file to ``False``. 

        Args:
            key (string): Name of parameter.

            vals (value or list of values): Allowed values of parameter.

            shorthand (string): Optional, shortened name of parameter. For 
                example, maybe the parameter ``steps_per_epoch`` is shortened
                to ``steps``. 

            in_name (bool): When constructing variant names, force the
                inclusion of this parameter into the name.
        """
        assert isinstance(key, str), "Key must be a string."
        assert shorthand is None or isinstance(shorthand, str), \
            "Shorthand must be a string."
        assert key not in self.keys , "You can't assign the key \"%s\" twice."%key
        if not isinstance(vals, list):
            vals = [vals]
        if DEFAULT_SHORTHAND and shorthand is None:
            shorthand = self._default_shorthand(key)
        self.keys.append(key)
        self.vals.append(vals)
        self.shs.append(shorthand)
        self.in_names.append(in_name)

    def variant_name(self, variant):
        """
        Given a variant (dict of valid param/value pairs), make an exp_name.

        A variant's name is constructed as the grid name (if you've given it 
        one), plus param names (or shorthands if available) and values 
        separated by underscores.

        Note: if ``seed`` is a parameter, it is not included in the name.
        """

        def get_val(v, k):
            # Utility method for getting the correct value out of a variant
            # given as a nested dict. Assumes that a parameter name, k, 
            # describes a path into the nested dict, such that k='a:b:c'
            # corresponds to value=variant['a']['b']['c']. Uses recursion
            # to get this.
            if k in v:
                return v[k]
            else:
                splits = k.split(':')
                k0, k1 = splits[0], ':'.join(splits[1:])
                return get_val(v[k0], k1)

        # Start the name off with the name of the variant generator.
        var_name = self._name

        # Build the rest of the name by looping through all parameters,
        # and deciding which ones need to go in there.
        for k, v, sh, inn in zip(self.keys, self.vals, self.shs, self.in_names):

            # Include a parameter in a name if either 
            # 1) it can take multiple values, or 
            # 2) the user specified that it must appear in the name.
            # Except, however, when the parameter is 'seed'. Seed is handled
            # differently so that runs of the same experiment, with different 
            # seeds, will be grouped by experiment name.
            if (len(v)>1 or inn) and not(k=='seed'):

                # Use the shorthand if available, otherwise the full name.
                param_name = sh if sh is not None else k
                param_name = valid_str(param_name)

                # Get variant value for parameter k
                variant_val = get_val(variant, k)

                # Append to name
                if all_bools(v): 
                    # If this is a param which only takes boolean values,
                    # only include in the name if it's True for this variant.
                    var_name += ('_' + param_name) if variant_val else ''
                else:
                    var_name += '_' + param_name + valid_str(variant_val)

        return var_name.lstrip('_')

    def _variants(self, keys, vals):
        """
        Recursively builds list of valid variants.
        """
        if len(keys)==1:
            pre_variants = [dict()]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                v = {}
                v[keys[0]] = val
                v.update(pre_v)
                variants.append(v)
        return variants

    def variants(self):
        """
        Makes a list of dicts, where each dict is a valid config in the grid.

        There is special handling for variant parameters whose names take
        the form

            ``'full:param:name'``.

        The colons are taken to indicate that these parameters should
        have a nested dict structure. eg, if there are two params,

            ====================  ===
            Key                   Val
            ====================  ===
            ``'base:param:a'``    1
            ``'base:param:b'``    2
            ====================  ===

        the variant dict will have the structure：

            variant = {
                base: {
                    param : {
                        a : 1,
                        b : 2
                        }
                    }    
                }
        """
        flat_variants = self._variants(self.keys, self.vals)

        # return flat_variants

        def unflatten_var(var):
            """ 
            Recursively build the full nested dict version of var, based on key names.
            """
            new_var = dict()
            unflatten_set = set()

            for k,v in var.items():
                if ':' in k:
                    splits = k.split(':')
                    k0 = splits[0]
                    assert k0 not in new_var or isinstance(new_var[k0], dict), \
                        "You can't assign multiple values to the same key \"%s\"."%k0

                    if not(k0 in new_var):
                        new_var[k0] = dict()

                    sub_k = ':'.join(splits[1:])
                    new_var[k0][sub_k] = v
                    unflatten_set.add(k0)
                else:
                    assert not(k in new_var), \
                        "You can't assign multiple values to the same key \"%s\"."%k
                    new_var[k] = v

            # Make sure to fill out the nested dicts.
            for k in unflatten_set:
                new_var[k] = unflatten_var(new_var[k])

            return new_var

        new_variants = [unflatten_var(var) for var in flat_variants]
        return new_variants

    def run(self, thunk, data_dir=None, datestamp=False):
        """
        Run each variant in the grid with function 'thunk'.

        Note: 'thunk' must be either a callable function, or a string. If it is
        a string, it must be the name of a parameter whose values are all 
        callable functions.

        Uses ``call_experiment`` to actually launch each experiment, and gives
        each variant a name using ``self.variant_name()``.

        Maintenance note: the args for ExperimentGrid.run should track closely
        to the args for call_experiment. However, ``seed`` is omitted because
        we presume the user may add it as a parameter in the grid.
        """

        # Print info about self.
        self.print()

        # Make the list of all variants.
        variants = self.variants()

        # Print variant names for the user.
        var_names = set([self.variant_name(var) for var in variants])
        var_names = sorted(list(var_names))
        line = '='*DIV_LINE_WIDTH
        preparing = colorize('Preparing to run the following experiments...', 
                                color='green', bold=True)
        joined_var_names = '\n'.join(var_names)
        announcement = f"\n{preparing}\n\n{joined_var_names}\n\n{line}"
        print(announcement)


        if WAIT_BEFORE_LAUNCH > 0:
            delay_msg = colorize(dedent("""
            Launch delayed to give you a few seconds to review your experiments.

            To customize or disable this behavior, change WAIT_BEFORE_LAUNCH in
            default.cfg and config.py.

            """), color='cyan', bold=True) + line
            print(delay_msg)
            wait, steps = WAIT_BEFORE_LAUNCH, 100
            prog_bar = trange(steps, desc='Launching in...', 
                              leave=False, ncols=DIV_LINE_WIDTH, 
                              mininterval=0.25,
                              bar_format='{desc}: {bar}| {remaining} {elapsed}')
            for _ in prog_bar:
                time.sleep(wait/steps)

        # Run the variants.
        for var in variants:
            exp_name = self.variant_name(var)

            # Figure out what the thunk is.
            if isinstance(thunk, str):
                # Assume one of the variant parameters has the same
                # name as the string you passed for thunk, and that 
                # variant[thunk] is a valid callable function.
                thunk_ = var[thunk]
                del var[thunk]
            else:
                # Assume thunk is given as a function.
                thunk_ = thunk

            self._call_experiment(exp_name, thunk_, data_dir=data_dir, datestamp=datestamp, **var)
    
        
    def _call_experiment(self, exp_name, thunk, data_dir=None, datestamp=False, **kwargs):
        """
        Run a function (thunk) with hyperparameters (kwargs), plus configuration.

        This wraps a few pieces of functionality which are useful when you want
        to run many experiments in sequence, including logger configuration and
        splitting into multiple processes for MPI.

        There's also a SpinningUp-specific convenience added into executing the
        thunk: if ``env_name`` is one of the kwargs passed to call_experiment, it's
        assumed that the thunk accepts an argument called ``env_fn``, and that
        the ``env_fn`` should make a gym environment with the given ``env_name``. 

        The way the experiment is actually executed is slightly complicated: the
        function is serialized to a string, and then ``run_entrypoint.py`` is
        executed in a subprocess call with the serialized string as an argument.
        ``run_entrypoint.py`` unserializes the function call and executes it.
        We choose to do it this way---instead of just calling the function 
        directly here---to avoid leaking state between successive experiments.

        Args:

            exp_name (string): Name for an experiment.
            thunk (callable): A python function.
            data_dir (string): Used in configuring the logger, to decide where
                to store experiment results. Note: if left as None, data_dir will
                default to ``DEFAULT_DATA_DIR`` from ``config.py``. 
            **kwargs: All kwargs to pass to thunk.

        """

        # Send exp_name, data_dir and datestamp to thunk
        kwargs['exp_name'] = exp_name
        kwargs['data_dir'] = data_dir
        kwargs['datestamp'] = datestamp

        # Be friendly and print out your kwargs, so we all know what's up
        print(colorize('Running experiment:\n', color='cyan', bold=True))
        print(exp_name + '\n')
        print(colorize('with kwargs:\n', color='cyan', bold=True))
        kwargs_json = convert_json(kwargs)
        print(json.dumps(kwargs_json, separators=(',',':\t'), indent=4, sort_keys=True))
        print('\n')

        # really invoking thunk to run
        thunk(**kwargs)

        # After experiments are finished, tell the user about where results are, and how to check them
        plot_cmd = colorize('python -m rltookit.run plot [output_dir]', color='green')
        test_cmd = colorize('python -m rltookit.run test_policy [output_dir]', color='green')
        output_msg = '\n'*5 + '='*DIV_LINE_WIDTH +'\n' + dedent("""\
        End of experiment.

        Plot results from this run with:

        %s

        Watch the trained agent with:

        %s

        """%(plot_cmd,test_cmd)) + '='*DIV_LINE_WIDTH + '\n'*5

        print(output_msg)


def parse_and_execute_grid_search(algo, args):
    """Interprets algorithm name and cmd line args into an ExperimentGrid."""

    # Before all else, check to see if any of the flags is 'help'.
    valid_help = ['--help', '-h', 'help']
    if any([arg in valid_help for arg in args]):
        print('\n\nShowing docstring for '+algo.__name__+':\n')
        print(algo.__doc__)
        sys.exit()

    def process(arg):
        # Process an arg by eval-ing it, so users can specify more
        # than just strings at the command line (eg allows for
        # users to give functions as args).
        try:
            return eval(arg)
        except:
            return arg

    # Make first pass through args to build base arg_dict. Anything
    # with a '--' in front of it is an argument flag and everything after,
    # until the next flag, is a possible value.
    arg_dict = dict()
    for i, arg in enumerate(args):
        assert i > 0 or '--' in arg, \
            friendly_err("You didn't specify a first flag.")
        if '--' in arg:
            arg_key = arg.lstrip('-')
            arg_dict[arg_key] = []
        else:
            arg_dict[arg_key].append(process(arg))

    # pass through, to catch flags that have no vals.
    # Assume such flags indicate that a boolean parameter should have
    # value True.
    # WARNING: "seed" can't be set to bool
    for k,v in arg_dict.items():
        if len(v) == 0:
            if k == "seed" or k == "s":
                v = v.append(DEFAULT_SEED)
            else:
                v.append(True)

    # Third pass: check for user-supplied shorthands, where a key has
    # the form --keyname{kn}. The thing in brackets, 'kn', is the
    # shorthand. NOTE: modifying a dict while looping through its
    # contents is dangerous, and breaks in 3.6+. We loop over a fixed list
    # of keys to avoid this issue.
    given_shorthands = dict()
    fixed_keys = list(arg_dict.keys())
    for k in fixed_keys:
        p1, p2 = k.find('{'), k.find('}')
        if p1 >= 0 and p2 >= 0:
            # Both '{' and '}' found, so shorthand has been given
            k_new = k[:p1]
            shorthand = k[p1+1:p2]
            given_shorthands[k_new] = shorthand
            arg_dict[k_new] = arg_dict[k]
            del arg_dict[k]

    # Penultimate pass: sugar. Allow some special shortcuts in arg naming,
    # eg treat "env" the same as "env_name". This is super specific
    # to Spinning Up implementations, and may be hard to maintain.
    # These special shortcuts are described by SUBSTITUTIONS.
    for special_name, true_name in SUBSTITUTIONS.items():
        if special_name in arg_dict:
            # swap it in arg dict
            arg_dict[true_name] = arg_dict[special_name]
            del arg_dict[special_name]

        if special_name in given_shorthands:
            # point the shortcut to the right name
            given_shorthands[true_name] = given_shorthands[special_name]
            del given_shorthands[special_name]

    # Final pass: check for the special args that go to the 'run' command
    # for an experiment grid, separate them from the arg dict, and make sure
    # that they have unique values. The special args are given by RUN_KEYS.
    run_kwargs = dict()
    for k in RUN_KEYS:
        if k in arg_dict:
            val = arg_dict[k]
            assert len(val) == 1, \
                friendly_err("You can only provide one value for %s."%k)
            run_kwargs[k] = val[0]
            del arg_dict[k]

    # Determine experiment name. If not given by user, will be determined
    # by the algorithm name.
    if 'exp_name' in arg_dict:
        assert len(arg_dict['exp_name']) == 1, \
            friendly_err("You can only provide one value for exp_name.")
        exp_name = arg_dict['exp_name'][0]
        del arg_dict['exp_name']
    else:
        exp_name = 'exp_' + algo.__name__

    # Add default seed if user didn't set it.
    if 'seed' not in arg_dict:
        warning_msg = colorize(dedent("""
            [--seed | -s] is not passed, DEFAULT_SEED=1 has been set.

            To customize or disable this behavior, change DEFAULT_SEED in
            default.cfg and config.py.

            """), color='cyan', bold=True)
        print(warning_msg)

        arg_dict['seed'] = DEFAULT_SEED

    assert len(arg_dict) >= 1, \
        friendly_err(dedent("""
        You have to pass at least one arg except [--exp_name | --datestamp | --datadir]
        to generate experiment varients, like [--seed] or any other args the running algo need.
        Add one and try again."""))

    # Construct and execute the experiment grid.
    eg = ExperimentGrid(name=exp_name)
    for k,v in arg_dict.items():
        eg.add(k, v, shorthand=given_shorthands.get(k))
    eg.run(algo, **run_kwargs)


def experiment_grid_test():
    from .algo_demo.dqn import dqn
    eg = ExperimentGrid('DqnExpGridTest')
    eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('seed', [1, 2])
    eg.add('batch_size', [32, 64])
    eg.add('gamma', [0.9, 0.95])
    eg.add('replay_size', 10000)

    # eg.run(dqn, data_dir='./rldata', datestamp=True)

    # L566 can be switched to following L569 and L570
    eg.add('algo', dqn)
    eg.run('algo', data_dir='./rldata', datestamp=True)


if __name__ == '__main__':
    """
    This is a wrapper allowing command-line interfaces to individual
    algorithms and the plot / test_policy utilities.

    For utilities, it only checks which thing to run, and calls the
    appropriate file, passing all arguments through.

    For algorithms, it sets up an ExperimentGrid object and uses the
    ExperimentGrid run routine to execute each possible experiment.
    """

    cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'

    algo = valid_cmd(cmd)

    valid_help = ['--help', '-h', 'help']

    if cmd in valid_help:
        # Before all else, check to see if any of the flags is 'help'.

        help_msg = dedent("""
            Running experiments from the command line with

            \tpython -m rltookit.run [algo] [experiment flags...]

            where CMD is a valid callable python object.
            """)

        print(help_msg)

        # Provide some useful details for algorithm running.
        subs_list = ['--' + k.ljust(10) + 'for'.ljust(10) + '--' + v \
                     for k,v in SUBSTITUTIONS.items()]
        str_valid_subs = '\n\t' + '\n\t'.join(subs_list)
        special_info = dedent("""
            FYI: When running an algorithm, any keyword argument to the
            algorithm function can be used as a flag, eg

            \tpython -m rltookit.run ppo --env HalfCheetah-v2 --clip_ratio 0.1

            If you need a quick refresher on valid kwargs, get the docstring
            with

            \tpython -m rltookit.run [algo] --help


            Also: Some common but long flags can be substituted for shorter
            ones. Valid substitutions are:
            """) + str_valid_subs
        print(special_info)

    elif algo:
        # Assume that the user plans to execute an algorithm. Run custom
        # parsing on the arguments and build a grid search to execute.
        args = sys.argv[2:]
        parse_and_execute_grid_search(algo, args)
    
    else:
        error_info = dedent("""
        Invalid cmd [%s], you can run an callable algorithm or look for help info with:
        
        python rltookit.run help or [--help | help].""")

        raise ValueError(error_info%cmd)
