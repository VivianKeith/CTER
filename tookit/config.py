"""
set up some necessary configuration
"""

from configparser import DEFAULTSECT
from textwrap import dedent
import os.path as osp
import json

# Command line args that will go to ExperimentGrid.run, and must possess unique
# values (therefore must be treated separately).
RUN_KEYS = ['data_dir', 'datestamp']

# Command line sweetener, allowing short-form flags for common, longer flags.
SUBSTITUTIONS = {'dt': 'datestamp',
                'exp': 'exp_name',
                'dir': 'data_dir',
                's': 'seed'}

# color value for print colorized message
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')
# Whether to automatically insert a date and time stamp into the names of save directories:
FORCE_DATESTAMP = False
# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True
# Default seed for random number generators.
DEFAULT_SEED = 1
# Tells the GridSearch how many seconds to pause for before launching experiments.
WAIT_BEFORE_LAUNCH = 3

def load_config_from_json(cfgpath):
    try:
        with open(cfgpath, 'r') as cfg:
            global DEFAULT_DATA_DIR, FORCE_DATESTAMP, DEFAULT_SHORTHAND, DEFAULT_SEED, WAIT_BEFORE_LAUNCH
            default_config = json.load(cfg)
            DEFAULT_DATA_DIR = default_config["default_data_dir"]
            FORCE_DATESTAMP = default_config["force_datestamp"]
            DEFAULT_SHORTHAND = default_config["default_shorthand"]
            DEFAULT_SEED = default_config["default_seed"]
            WAIT_BEFORE_LAUNCH = default_config["wait_before_launch"]
    except:
        error_info = dedent("""
        [WARNING] Missing default config or it is not complete.
        
        Now, the default config has been set as:
        {
            DEFAULT_DATA_DIR: 'parentdir/of/rltookit.config/data',
            DEFAULT_SEED: 1,
            DEFAULT_SHORTHAND": True,
            FORCE_DATESTAMP: False,
            WAIT_BEFORE_LAUNCH: 5
        }

        You can set default config with:
        
        python -m rltookit.config [-h | --help] [--default_data_dir]
                        [--force_datestamp] [--default_shorthand]
                        [--default_seed] [--wait_before_launch]
                        [--restore | -r]

        """)
        print(error_info)


def save_config_to_json(
    default_data_dir,
    force_datestamp,
    default_shorthand,
    default_seed,
    wait_before_launch,
):
    output = json.dumps(locals(), separators=(',',':\t'), indent=4, sort_keys=True)
    cfgpath = osp.join(osp.abspath(osp.dirname(__file__)),'default.cfg')
    print('Saving default config to {}:\n'.format(cfgpath))
    print(output)
    with open(cfgpath, 'w') as out:
        out.write(output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="rltookit.config", description="Set default configurations for rltookit. ")

    parser.add_argument('--default_data_dir', '-ddd', metavar='',
                        default=osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data'),
                        help="Where experiment outputs are saved by default.")
    parser.add_argument('--force_datestamp', '-fd', action='store_true',
                        help="Whether to automatically insert a date and time stamp into the names of save directories.")
    parser.add_argument('--default_shorthand', '-dsh', action='store_true',
                        help="Whether GridSearch provides automatically-generated default shorthands.")
    parser.add_argument('--default_seed', '-dse', type=int, default=1, metavar='',
                        help="Default seed for random number generators.")
    parser.add_argument('--wait_before_launch', '-wbl', type=int, default=5, metavar='',
                        help="Tells the GridSearch how many seconds to pause for before launching experiments.")
    parser.add_argument('--restore', '-r', action='store_true',
                        help="Restore all default configs.")
    args = parser.parse_args()

    if args.restore:
        save_config_to_json(
            default_data_dir=DEFAULT_DATA_DIR,
            force_datestamp=FORCE_DATESTAMP,
            default_shorthand=DEFAULT_SHORTHAND,
            default_seed=DEFAULT_SEED,
            wait_before_launch=WAIT_BEFORE_LAUNCH
        )
    else:
        save_config_to_json(
            default_data_dir=osp.abspath(args.default_data_dir),
            force_datestamp=args.force_datestamp,
            default_shorthand=args.default_shorthand,
            default_seed=args.default_seed,
            wait_before_launch=args.wait_before_launch
        )
else:
    load_config_from_json(osp.join(osp.abspath(osp.dirname(__file__)),'default.cfg'))