"""
Contains all the parameter values for one simulation.
"""

import argparse

parser = argparse.ArgumentParser()

# General system parameters
parser.add_argument("--width",
                    type=int,
                    default=70,
                    help="Integer value representing the number of grid cells")
parser.add_argument("--height",
                    type=int,
                    default=70,
                    help="Integer value representing the number of grid cells")
parser.add_argument(
    "--torus",
    action="store_true",
    default=False,
    help="Create grid as torus if passed, bounded if not specified")
parser.add_argument("--household_density",
                    type=float,
                    default=0.9,
                    help="Total fraction of grid cells occupied by households")
parser.add_argument("--student_density",
                    type=float,
                    default=1,
                    help="Average amount of students per household")
parser.add_argument(
    "--max_move_fraction",
    type=float,
    default=0.125,
    help="Maximum fraction of agents allowed to move during a full step")
parser.add_argument("--max_res_steps",
                    type=int,
                    default=0,
                    help="Maximum amount of steps during residential process")
parser.add_argument("--max_school_steps",
                    type=int,
                    default=200,
                    help="Maximum amount of steps during school process")
parser.add_argument(
    "--conv_threshold",
    type=float,
    default=0.01,
    help="Convergence threshold for the residential and school processes")

# Spatial object parameters
parser.add_argument("--n_neighbourhoods",
                    type=int,
                    default=49,
                    help="Number of neighbourhood objects to be placed")
parser.add_argument(
    "--neighbourhoods_placement",
    type=str,
    default="evenly_spaced",
    help="Placement method of neighbourhoods: random, evenly_spaced")
parser.add_argument("--n_schools",
                    type=int,
                    default=49,
                    help="Amount of schools to be place in grid")
parser.add_argument(
    "--schools_placement",
    type=str,
    default="evenly_spaced",
    help=
    "Placement method of schools: random, evenly_spaced, random_per_neighbourhood"
)

# Household parameters
parser.add_argument("--group_categories",
                    type=str,
                    nargs="+",
                    default=["etnicity"],
                    help="Space seperated list of categories")
parser.add_argument(
    "--group_types",
    type=str,
    nargs="+",
    default=[["a", "b"]],
    help=
    "Space seperated list of strings containing comma seperated list of types within each group"
)
parser.add_argument(
    "--group_dist",
    type=str,
    nargs="+",
    default=[[0.5, 0.5]],
    help=
    "Space seperated list of strings containing comma seperated list of density of subtype"
)
parser.add_argument(
    "--utility_at_max",
    type=str,
    nargs="+",
    default=[[0.6, 0.6]],
    help=
    "Space seperated list of strings containing comma seperated list of density of subtype"
)
parser.add_argument(
    "--optimal_fraction",
    type=str,
    nargs="+",
    default=[[0.5, 0.5]],
    help=
    "Space seperated list of strings containing comma seperated list of density of subtype"
)

# School parameters
parser.add_argument(
    "--school_capacity",
    type=float,
    default=2,
    help="Capacity of school as fraction of students in encatchment area")

# Parameters determining which neighbours are used
parser.add_argument(
    "--radius",
    type=int,
    default=3,
    help="Radius of circle over which neighbours are taken into account. ")
parser.add_argument("--moore",
                    dest="moore",
                    action="store_true",
                    help="Use Moore neighbours when finding neighbours")
parser.add_argument("--no-moore",
                    dest="moore",
                    action="store_false",
                    help="Do not use Moore neighbours when finding neighbours")
parser.set_defaults(moore=True)
parser.add_argument(
    "--neighbourhood_mixture",
    type=float,
    default=0.2,
    help=
    "Ratio bounded neighbourhood to local neighbourhood used in Schelling model. \
                            Zero equals fully local, one fully bounded neighbourhood."
)

# Other parameters
parser.add_argument("--alpha",
                    type=float,
                    default=0.25,
                    help="Weight for composition and distance utility")
parser.add_argument("--filename",
                    type=str,
                    default='data/data',
                    help="Filename to save the data.")
parser.add_argument(
    "--temperature",
    type=float,
    default=50,
    help="Controls the random versus deterministic part in the \
                        behavioural logit rule.")
parser.add_argument("--homophily_std",
                    type=float,
                    default=0.02,
                    help="Standard deviation of the truncated normal \
                    distribution used in sampling the homphily thresholds.")
parser.add_argument("--num_considered",
                    type=int,
                    default=1,
                    help="How many empty spots an agent considers every \
                        residential move.")
parser.add_argument("--window_size",
                    type=int,
                    default=30,
                    help="How many time steps one looks back for convergence.")
parser.add_argument("--ranking_method",
                    type=str,
                    default='proportional',
                    help="The ranking method of the empty residential spots, \
                    one of 'highest' or 'proportional'")
parser.add_argument("--scheduling",
                    type=str,
                    default=1,
                    help="Does not replace the agent if the value is \
                    1 ('without_replacement'), 0 is with replacement. \
                    Agents are randomly shuffled again \
                    after all have had a chance to move.")
parser.add_argument(
    "--case",
    type=str,
    default='Amsterdam',
    help="To implement case studies. Only lattice and Amsterdam \
                        available for now.")
parser.add_argument("--random_residential",
                    type=int,
                    default=0,
                    help="To generate random residential patterns.")
parser.add_argument("--verbose",
                    type=bool,
                    default=False,
                    help="To enable print statements or not.")
parser.add_argument("--save_last_only",
                    type=bool,
                    default=False,
                    help="Only save last time step to save space on disk.")
parser.add_argument(
    "--p",
    type=int,
    default=4000,
    help="Controls the location of the 0.5 utility for the distance sigmoid")
parser.add_argument("--q",
                    type=int,
                    default=2,
                    help="Controls the slope of the distance sigmoid")

# Parse and reformat if necessary (multiple group categories)
FLAGS, unparsed = parser.parse_known_args()
if type(FLAGS.group_types[0]) == str:
    FLAGS.group_types = [elem.split(",") for elem in FLAGS.group_types]
    FLAGS.group_dist = [elem.split(",") for elem in FLAGS.group_dist]
    FLAGS.group_dist = [[float(element) for element in sublist]
                        for sublist in FLAGS.group_dist]
    FLAGS.homophilies = [elem.split(",") for elem in FLAGS.homophilies]
    FLAGS.homophilies = [[float(element) for element in sublist]
                         for sublist in FLAGS.homophilies]
    FLAGS.utility_at_max = [elem.split(",") for elem in FLAGS.utility_at_max]
    FLAGS.utility_at_max = [[float(element) for element in sublist]
                            for sublist in FLAGS.utility_at_max]
    FLAGS.optimal_fraction = [
        elem.split(",") for elem in FLAGS.optimal_fraction
    ]
    FLAGS.optimal_fraction = [[float(element) for element in sublist]
                              for sublist in FLAGS.optimal_fraction]
