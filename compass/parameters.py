"""
Contains all the parameter values for one simulation.
"""

import argparse

parser = argparse.ArgumentParser()

# General system parameters
parser.add_argument("--width",
                    type=int,
                    default=50,
                    help="Integer value representing the number of grid cells (lattice case only)")
parser.add_argument("--height",
                    type=int,
                    default=50,
                    help="Integer value representing the number of grid cells (lattice case only)")
parser.add_argument("--torus",
                    action="store_true",
                    default=True,
                    help="Create grid as torus if passed, bounded if not specified (lattice case only)")
parser.add_argument("--household_density",
                    type=float,
                    default=0.9,
                    help="Total fraction of grid cells occupied by households (lattice case only)")
parser.add_argument("--student_density",
                    type=float,
                    default=1,
                    help="Average amount of students per household")
parser.add_argument("--max_move_fraction",
                    type=float,
                    default=0.125,
                    help="Maximum fraction of agents allowed to move during a full step")
parser.add_argument("--max_res_steps",
                    type=int,
                    default=100,
                    help="Maximum amount of steps during residential process (lattice case only)")
parser.add_argument("--max_school_steps",
                    type=int,
                    default=200,
                    help="Maximum amount of steps during school process")
parser.add_argument("--conv_threshold",
                    type=float,
                    default=0.01,
                    help="Convergence threshold for the residential and school processes")
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
parser.add_argument("--case",
                    type=str,
                    default='Amsterdam-ses',
                    help="To implement case studies. Only lattice and several Amsterdam \
                            cases available for now.")
parser.add_argument("--verbose",
                    type=bool,
                    default=False,
                    help="To enable print statements or not.")
parser.add_argument("--save_last_only",
                    type=bool,
                    default=True,
                    help="Only save last time step to save space on disk.")
parser.add_argument("--filename",
                    type=str,
                    default='data/data',
                    help="Filename to save the data.")
parser.add_argument("--temperature",
                    type=float,
                    default=50,
                    help="Controls the random versus deterministic part in the \
                            behavioural logit rule for all households.")

# Spatial object parameters
parser.add_argument("--n_neighbourhoods",
                    type=int,
                    default=25,
                    help="Number of neighbourhood objects to be placed (lattice case only)")
parser.add_argument("--n_schools",
                    type=int,
                    default=25,
                    help="Amount of schools to be place in grid (lattice case only)")
parser.add_argument("--schools_placement",
                    type=str,
                    default="evenly_spaced",
                    help=
                    "Placement method of schools: random, evenly_spaced, random_per_neighbourhood (lattice case only)")

# Household parameters
parser.add_argument("--group_categories",
                    type=str,
                    nargs="+",
                    default=["attr1"],
                    help="Space seperated list of household groups (e.g. ethnicity, income)")
parser.add_argument("--group_types",
                    type=str,
                    nargs="+",
                    default=[["a", "b"]],
                    help=
                    "Space seperated list of strings containing comma seperated list of types within each group")
parser.add_argument("--group_dist",
                    type=str,
                    nargs="+",
                    default=[[0.5, 0.5]],
                    help=
                    "Space seperated list of strings containing comma seperated list of density of subtype")
parser.add_argument("--utility_at_max",
                    type=str,
                    nargs="+",
                    default=[[0.8, 0.8]],
                    help=
                    "Space seperated list of strings containing comma seperated list of density of subtype")
parser.add_argument("--optimal_fraction",
                    type=str,
                    nargs="+",
                    default=[[0.6, 0.4]],
                    help=
                    "Space seperated list of strings containing comma seperated list of density of subtype")
parser.add_argument("--alpha",
                    type=float,
                    default=[[0.5, 0.3]],
                    help="Weight for composition and distance utility")
parser.add_argument("--p",
                    type=int,
                    default=[[2000, 1000]],
                    help="Controls the location of the 0.5 utility for the distance sigmoid")
parser.add_argument("--q",
                    type=int,
                    default=[[2, 4]],
                    help="Controls the slope of the distance sigmoid")
parser.add_argument("--stdev",
                    type=float,
                    default=0.01,
                    help="Standard deviation of the truncated normal \
                    distribution used in sampling.")
parser.add_argument("--num_considered",
                    type=int,
                    default=8,
                    help="How many empty spots a household considers every \
                        residential move.")
parser.add_argument("--radius",
                    type=int,
                    default=1,
                    help="Radius of circle over which neighbours are taken into account (lattice case only).")
parser.add_argument("--neighbourhood_mixture",
                    type=float,
                    default=0,
                    help=
                    "Ratio bounded neighbourhood to local neighbourhood used in Schelling model (lattice case only). \
                        Zero equals fully local, one fully bounded neighbourhood.")

# School parameters
parser.add_argument("--school_capacity",
                    type=float,
                    default=2,
                    help="Capacity of school as fraction of students in encatchment area")
parser.add_argument("--min_capacity",
                    type=int,
                    default=50,
                    help="Minimum capacity of schools.")

# Parameters for experiments
parser.add_argument("--random_residential",
                    type=int,
                    default=0,
                    help="To generate random residential patterns.")
parser.add_argument("--subset_schools",
                    type=int,
                    default=0,
                    help="To only consider a subset of schools (equal to num_considered).")
parser.add_argument("--visualisation",
                    type=bool,
                    default=False,
                    help="Whether to gather data for visualisation (used in notebooks)")

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
