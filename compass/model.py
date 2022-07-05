"""
The Model class which initialises the system and all of its components.
"""
import contextlib
from datetime import datetime
import os
import sys
from typing import List, Dict, Generator
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
import ijson
import pandas as pd
import geopandas as gpd
from scipy.ndimage import convolve
from scipy.stats import truncnorm
from shapely.geometry import Point, box
from .household import Household
from .school import School
from .neighbourhood import Neighbourhood
from .scheduler import ThreeStagedActivation
from .utils import Measurements
from .functions import calc_comp_utility


@contextlib.contextmanager
def record_time(name: str) -> Generator:
    try:
        start_time = datetime.now()
        yield
    finally:
        print("\n%s: %f" % (name, (datetime.now() - start_time).total_seconds()))


def read_households(path: str) -> pd.DataFrame:
    """
    A fast and efficient parser to read household data from a GeoJSON file.
    Arguments
        path    File to open, passed directly to open()
    Returns
        households (pd.DataFrame) [pos, category, neighbourhood_id]
    """

    # open in binary mode for ijson
    geofile = open(path, "rb")

    # assume floats are good enough,
    # no need for aribrary precision Decimal objects
    geo_households = ijson.items(geofile, "features.item", use_float=True)

    # normal python lists, because appending is fast for them,
    # and we need the columns to create a pandas DataFrame
    coordinates = []
    groups = []
    neighbourhood_ids = []

    for feature in geo_households:
        coordinates.append(feature["geometry"]["coordinates"])
        groups.append(feature["properties"]["group"])
        neighbourhood_ids.append(feature["properties"]["neighbourhood_id"])
    geofile.close()

    return pd.DataFrame(
        {"pos": coordinates, "category": groups, "neighbourhood_id": neighbourhood_ids}
    )


def trunc_normal_sample(means: List, scale: float, size: int, seed=None) -> List[np.ndarray]:
    """
    Samples from a set of truncated normal distributions.
    Arguments:
        means  [n_means] of float
        scale  float
        size   float
    Output:
        sample [n_means] of np.ndarray [size]
    """
    truncnorm._random_state = seed

    sample = [0] * len(means)
    for index, mu in enumerate(means):
        if scale == 0:
            # All samples are equal if the scale is zero.
            sample[index] = np.ones(size) * mu
        else:
            sample[index] = truncnorm.rvs(
                (0 - mu) / scale, (1 - mu) / scale, loc=mu, scale=scale, size=size
            )
    return sample


class CompassModel(Model):
    """
    Model class for school segregation dynamics.
    Args:
        params (Argparser): containing all parameter values.
    Attributes:
        params (dict):
        grid (MultiGrid): MultiGrid object from Mesa.
        scheduler (ThreeStagedActivation): ThreeStagedActivation object.
        agents (list): all agents in the model.
        measurements (Measurements): Measurements object.
        global_composition (dict): containing the total system compositions.
        distance_matrix (list): all the Euclidean distances from one grid cell
            to another.
        nearness_matrix (list): same as above but with normalized distances.
        alpha np.ndarray[n_households]
        temperature float
        utility_at_max np.ndarray[n_households]
        optimal_fraction np.ndarray [n_households]
        neighbourhood_mixture np.ndarray [n_households]
        chosen_indices np.ndarray [n_households]
        school_ended boolean
        res_ended boolean
    Cached attributes:
        kernel (ndarray)
            set: calc_residential_compositions
            used: calc_residential_compositions
        local_compositions (list, ndarray) [n_households, ndarray]
            set: set_agent_parameters, overwritten in load_agents, update_residential
            used: calc_res_utilities
        normalized_compositions (ndarray) [width, height, len(params['group_types'][0])]
            set: calc_residential_compositions
            used: set_agent_parameters
        neighbourhood_compositions (list, ndarray) [n_households]
            set: set_agent_parameters, update_residential
            used: load_agents, calc_res_utilities
        school_compositions (ndarray, float32) [n_households]
            set: set_agent_parameters, calc_school_compositions, update_school
            used: calc_school_utilities
        alpha (ndarray, float32) [n_households]
            set: set_agent_parameters
            used: calc_school_utilities, calc_school_rankings
        temperature (float?)
            set: set_agent_parameters
            used: (calc_school_rankings, residential_ranking but from params)
        utility_at_max (ndarray, float32) [n_households]
            set: set_agent_parameters
            used: calc_res_utilities, calc_school_utilities , calc_school_rankings
            (residential_utility, but from params)
        optimal_fraction (ndarray, float32) [n_households]
            set: set_agent_parameters
            used: calc_res_utilities, calc_school_utilities, calc_school_rankings
        neighbourhood_mixture (ndarray, int) [n_households]  TODO: np.ushort?
            set: set_agent_parameters
            used: calc_res_utilities
        distance_utilities(ndarray) [n_households, n_schools]
            set: set_agent_parameters
            used: calc_school_rankings, update_school
        all_distances:  (ndarray) [n_households, n_schools] FIXME
        household_attrs (ndarray, float) [width, height, len(params['group_types'][0])]
        compositions (ndarray, float32)  as household_attrs
    """

    def __init__(self, export: bool = False, **params: dict):

        super().__init__()

        # Use random from the numpy package for extra functionality
        self.random = np.random.RandomState(self._seed)

        # define various arrays TODO: sort/document those
        self.local_compositions: List = []
        self.neighbourhood_compositions: List = []
        self.alpha: np.ndarray = np.ndarray([])
        self.temperature: float = 0.0
        self.utility_at_max: np.ndarray = np.ndarray([])
        self.optimal_fraction: np.ndarray = np.ndarray([])
        self.neighbourhood_mixture: np.ndarray = np.ndarray([])
        self.school_compositions: np.ndarray = np.ndarray([])
        self.school_ended: bool = False
        self.res_ended: bool = False
        self.distance_utilities: np.ndarray = np.ndarray([])
        self.chosen_indices: np.ndarray = np.ndarray([])

        self.household_attrs: np.ndarray = np.ndarray([])
        self.compositions: np.ndarray = np.ndarray([])
        self.normalized_compositions: np.ndarray = np.ndarray([])

        self.all_distances: np.ndarray = np.ndarray([])
        self.kernel: np.ndarray = np.ndarray([])

        # Initialise the model attributes
        self.set_attributes(params=params, export=export)

        if self.params["case"].lower() != "lattice":
            self.load_agents(self.params["case"].lower())
        else:
            self.create_agents()

        # Get values of the initial configuration
        self.measurements.end_step(residential=True)

        # Calculate global compositions for the segregation calculations
        self.global_composition = self.measurements.neighbourhoods[0, :, :2].sum(axis=0)

        if self.verbose:
            text = f""" Model initialised:
                NR AGENTS:
                Households: {self.params['n_households']}
                Neighbourhoods: {self.params['n_neighbourhoods']}
                Schools: {self.params['n_schools']}
                In scheduler: {self.scheduler.get_agent_count()}"""
            print(text)

    def set_attributes(self, params: dict, export=False) -> None:
        """
        Sets or calculates all attributes used in the Compass class.
        Args:
            params (Argparser): containing all parameter values.
            export (bool): True if the data needs to be exported or not.
        """

        # Calculate number of households and students
        params["n_households"] = int(
            params["household_density"]
            * (
                params["width"] * params["height"]
                - params["n_neighbourhoods"]
                - params["n_schools"]
            )
        )
        params["n_students"] = int(params["n_households"] * params["student_density"])
        self.params = dict(params)

        # Set tracking attributes
        self.export = export
        # Track segregation over time
        self.segregation: List[float] = []
        self.res_ended = False
        self.school_ended = False
        self.verbose = self.params["verbose"]
        self.agents: Dict[str, List[object]] = {
            "amount": 0,
            "households": [],
            "schools": [],
            "neighbourhoods": [],
        }

        # Initialise other objects
        self.measurements = Measurements(self)
        self.scheduler = ThreeStagedActivation(self)
        self.grid = ContinuousSpace(
            self.params["width"], self.params["height"], torus=self.params["torus"]
        )

    def create_agents(self) -> None:
        """
        Creates the agents when no case study is provided.
        """
        self.neighbourhoods()
        self.schools()
        self.location_to_agent()

        # Compute closest neighbourhoods
        self.closest_neighbourhoods = self.compute_closest_neighbourhoods()

        # Create households
        self.households()

    def set_agent_parameters(self, params: dict) -> None:
        """
        Puts the agent parameters in numpy arrays for faster computations.
        Args:
            params (dict): Model parameters which could differ from
                the agent params!
        Todo:
            Parameters should be imported from a config file in the future.
        """
        households = self.get_agents("households")
        n_households = len(households)

        dtype = "float32"
        self.alpha = np.zeros(n_households, dtype=dtype)
        self.temperature = self.params[
            "temperature"
        ]  # FIXME: different temp per agent?
        self.utility_at_max = np.zeros(n_households, dtype=dtype)
        self.optimal_fraction = np.zeros(n_households, dtype=dtype)
        self.neighbourhood_mixture = np.ones(n_households, dtype=int)

        optimal_fractions = trunc_normal_sample(
            params["optimal_fraction"][0],
            scale=params["homophily_std"],
            size=n_households,
            seed=self.random
        )
        alphas = trunc_normal_sample(
            [params["alpha"], params["alpha"]],
            scale=params["homophily_std"],
            size=n_households,
            seed=self.random
        )
        utility_at_maxs = trunc_normal_sample(
            params["utility_at_max"][0],
            scale=params["homophily_std"],
            size=n_households,
            seed=self.random
        )

        for household in households:
            x, y = household.pos

            # Fill arrays with agent parameter values for faster computations
            self.optimal_fraction[household.idx] = optimal_fractions[
                household.category
            ][household.idx]
            self.alpha[household.idx] = alphas[household.category][household.idx]
            self.utility_at_max[household.idx] = utility_at_maxs[household.category][
                household.idx
            ]

            # Currently only convolution (assumes every household has the same
            # radius) for composition calculations within the lattice case.
            if params["case"].lower() == "lattice":
                self.local_compositions.append(
                    self.normalized_compositions[x, y, household.category]
                )
            else:
                household.params["neighbourhood_mixture"] = 1

            if household.neighbourhood.total > 0:
                norm = 1.0 / household.neighbourhood.total
            else:
                norm = 1.0
            self.neighbourhood_compositions.append(
                household.neighbourhood.composition[household.category] * norm
            )

        # These are filled with the actual distance and composition utilities
        # of the household and the school (singular!) they attend

        # SHOULD BE CALLED DIFFERENTLY CAUSE NOW IT OVERWRITES AN ATTRIBUTE!!!
        self.school_compositions = np.zeros(n_households, dtype=dtype)

        # Distance utilities based on sigmoid function
        if self.params["case"].lower() != "lattice":
            p = self.params["p"]
            q = self.params["q"]
            self.distance_utilities = 1.0 / (1 + (self.all_distances / p) ** q)

    def neighbourhoods(self) -> None:
        """
        Adds the neighbourhood objects to the environment.
        """

        n_neighs = self.params["n_neighbourhoods"]

        # Add neighbourhoods if necessary
        if n_neighs:
            locations = self.choose_locations(
                n_neighs, self.params["neighbourhoods_placement"]
            )
            for location in locations:
                size = self.params["width"] / float(n_neighs**0.5 * 2)
                minx, miny = location[0] - size, location[1] - size
                maxx, maxy = location[0] + size, location[1] + size
                shape = box(minx, miny, maxx, maxy)

                # Create the Neighbourhood object and place it on the grid and
                # add it to the scheduler
                neighbourhood = Neighbourhood(
                    self.get_agents("amount"), location, shape, self, self.params
                )
                self.agents["neighbourhoods"].append(neighbourhood)
                self.scheduler.add(neighbourhood)
                self.grid.place_agent(neighbourhood, location)

    def schools(self) -> None:
        """
        Adds the school objects to the environment.
        """

        School.reset()

        # Add schools if necessary
        if self.params["n_schools"]:
            locations = self.choose_locations(
                self.params["n_schools"], self.params["schools_placement"]
            )
            for location in locations:
                # Create the School object and place it on the grid and add it
                # to the scheduler
                school = School(self.get_agents("amount"), location, self, self.params)
                self.agents["schools"].append(school)
                self.scheduler.add(school)
                self.grid.place_agent(school, location)

    def households(self) -> None:
        """
        Adds the household objects to the environment.
        Todo:
            * Place household specific parameters in a attribute called params
        """

        params = self.params
        self.chosen_indices = None  # Only matters for case studies
        self.household_attrs = np.zeros(
            shape=(params["width"], params["height"], len(params["group_types"][0])),
            dtype="float32",
        )

        # Create group types, empty spots and shuffle them both
        n_groups = len(params["group_categories"])
        groups = [
            self.random.choice(
                list(range(0, len(params["group_types"][i]))),
                size=params["n_households"],
                p=params["group_dist"][i],
            )
            for i in range(n_groups)
        ]

        self.grid.empties = set(
            [(x, y) for x in range(params["width"]) for y in range(params["height"])]
        )
        empties = list(self.grid.empties)
        self.random.shuffle(empties)

        # pre-allocate storage for the Housholds
        Household.reset(max_households=params["n_households"])

        for i, position in enumerate(empties[0 : params["n_households"]]):
            household = Household(
                self.get_agents("amount"), position, self, params, groups[0][i]
            )

            # Place households on the grid and add them to the scheduler
            self.agents["households"].append(household)
            self.grid.place_agent(household, position)
            self.grid.empties.discard(position)
            self.scheduler.add(household)
            self.household_attrs[position[0], position[1], :] = household.attributes

        # Calculate AFTER all agents are placed
        self.calc_residential_compositions()
        self.set_agent_parameters(params)
        self.calc_res_utilities()

    def load_agents(self, case="Amsterdam") -> None:
        """
        Load the agents from several files.
        Note:
            This function is in progress and works only for the
            Amsterdam case now.
        """

        dirname = os.path.dirname(__file__)
        if case.lower() == "amsterdam":
            path = dirname + "/maps/amsterdam"
        elif case.lower() == "south-london":
            path = dirname + "/maps/south-london"
        elif case.lower() == "london":
            path = dirname + "/maps/london"

        # Load GeoDataFrames
        school_frame = gpd.read_file(path + "/schools.geojson")
        household_frame = read_households(path + "/households.geojson")
        neighbourhood_frame = gpd.read_file(path + "/neighbourhoods.geojson")

        # Create grid
        self.params["torus"] = 0
        self.params["max_res_steps"] = 0
        xmin, ymin, xmax, ymax = neighbourhood_frame.total_bounds
        self.grid = ContinuousSpace(xmax, ymax, self.params["torus"], xmin, ymin)
        self.grid.empties = [(0, 0)]

        # In the file more households could be available to sample from,
        # but only use the actual amount
        data = np.load(path + "/distances_perc_of_actual.npz")
        perc_of_actual = data["perc_of_actual"]
        self.all_distances = data["distances"]

        self.scheduler = ThreeStagedActivation(self)

        # More agents are simulated to sample from them and incorporate some
        # randomness in the type and spatial distribution
        total_households = len(household_frame)
        actual_households = int(total_households / perc_of_actual)
        self.params["n_households"] = actual_households
        self.params["n_students"] = int(
            self.params["n_households"] * self.params["student_density"]
        )

        # Create neighbourhoods
        self.create_neighbourhoods(neighbourhood_frame)

        # Create schools
        self.create_schools(school_frame)

        # Create households
        self.create_households(household_frame, actual_households)

        if self.verbose:
            print("Setting agent parameters...")
        self.set_agent_parameters(self.params)

        self.local_compositions = self.neighbourhood_compositions
        self.calc_res_utilities()

        if self.verbose:
            print("Model loaded!")

    def create_neighbourhoods(self, neighbourhood_frame: pd.DataFrame) -> None:
        """
        Given a GeoDataFrame, this creates all the neighbourhood objects
        """
        if self.verbose:
            print("Creating neighbourhoods...")

        self.params["n_neighbourhoods"] = len(neighbourhood_frame)
        for index, row in neighbourhood_frame.iterrows():
            neighbourhood = Neighbourhood(
                unique_id=index,
                pos=(row.geometry.centroid.xy[0][0], row.geometry.centroid.xy[1][0]),
                shape=row.geometry,
                model=self,
                params=self.params,
            )
            self.agents["neighbourhoods"].append(neighbourhood)
            self.scheduler.add(neighbourhood)
            self.grid.place_agent(neighbourhood, neighbourhood.pos)

    def create_schools(self, school_frame: pd.DataFrame) -> None:
        """
        Given a GeoDataFrame, this creates all the school objects
        """

        School.reset()

        if self.verbose:
            print("Creating schools...")

        self.params["n_schools"] = len(school_frame)
        n_neighbourhoods = self.params["n_neighbourhoods"]
        for index, row in school_frame.iterrows():
            school = School(
                unique_id=index + n_neighbourhoods,
                pos=(row.geometry.xy[0][0], row.geometry.xy[1][0]),
                model=self,
                params=self.params,
            )
            school.capacity = 1 + int(
                self.params["school_capacity"]
                * self.params["n_students"]
                / self.params["n_schools"]
            )
            self.agents["schools"].append(school)
            self.scheduler.add(school)
            self.grid.place_agent(school, school.pos)

    def create_households(
        self, household_frame: pd.DataFrame, actual_households: int
    ) -> None:
        """
        Given a GeoDataFrame, this creates household objects by randomly
        drawing (without replacement) household metadata rows from the frame.
        Arguments
            actual_households (int)
                The number of households to create.
        Parameters used
            params['random_residential']
        Parameters set
            params['n_households']
        """
        if self.verbose:
            print("Creating households...")

        self.chosen_indices = self.random.choice(
            len(household_frame), size=actual_households, replace=False
        )
        households_sample = household_frame.iloc[self.chosen_indices]

        if self.params["random_residential"]:
            # Randomly shuffle the group of the household
            shuffled = households_sample["group"].values
            self.random.shuffle(shuffled)
            households_sample["group"] = shuffled

        self.params["n_households"] = actual_households
        n_agents = self.params["n_neighbourhoods"] + self.params["n_schools"]
        neighbourhoods = self.get_agents("neighbourhoods")

        self.all_distances = self.all_distances[self.chosen_indices, :]

        # pre-allocate storage for the Housholds
        Household.reset(max_households=actual_households)

        # DataFrame.iterrows() is very slow, dont use it!
        # the current code uses only iterates (no copies)
        # and should be memory-friendly and performant:
        # accumulated time for create_households:
        # for the testcase with  60K households from 9.10 to 2.60 seconds
        # for the testcase with 210K households from 30.0 to 9.27 seconds
        for index, row in enumerate(
            zip(
                households_sample["pos"],
                households_sample["category"],
                households_sample["neighbourhood_id"],
            )
        ):
            household = Household(
                unique_id=index + n_agents,
                pos=row[0],
                model=self,
                params=self.params,
                category=row[1],
                nhood=neighbourhoods[row[2]],
            )
            self.agents["households"].append(household)
            self.scheduler.add(household)
            self.grid.place_agent(household, household.pos)

        self.location_to_agent()

    def location_to_agent(self) -> None:
        """
        Creates a dictionary with the location of the neighbourhoods as key and
        the object itself as value. Schools are not included as they can have
        the same position as a neighbourhood (centroid).
        """
        agents = self.get_agents("neighbourhoods")
        self.location_to_agent = {str(agent.pos): agent for agent in agents}

    def calc_residential_compositions(self) -> None:
        """
        Updates all local residential compositions assuming all households have
        the SAME RADIUS.
        """

        # Determine the kernel of the convolution
        radius = self.params["radius"]
        dim = radius * 2 + 1
        self.kernel = np.ones((dim, dim))
        self.kernel[radius, radius] = 0

        # Should it wrap around the edges or not?
        if self.params["torus"]:
            mode = "wrap"
        else:
            mode = "constant"

        summed = 0
        num_attrs = self.household_attrs.shape[2]
        compositions = np.zeros(shape=self.household_attrs.shape, dtype="float32")

        # Convolution for every household attribute.
        for attr in range(num_attrs):
            compositions[:, :, attr] = convolve(
                self.household_attrs[:, :, attr], self.kernel, mode=mode
            )
            summed += compositions[:, :, attr]
        self.compositions = compositions
        self.normalized_compositions = np.nan_to_num(
            compositions / np.repeat(summed[:, :, np.newaxis], num_attrs, axis=2)
        )

    def calc_school_compositions(self) -> None:
        """
        Calculate the new school compositions for every household and only for
        the first student!
        Note:
            Currently only for the first student!!!
        """

        households = self.get_agents("households")
        for household in households:
            category = household.category
            idx = household.idx
            school = household.school
            if school.total:
                self.school_compositions[idx] = (
                    school.composition[category] / school.total
                )
            else:
                self.school_compositions[idx] = 0.0

    def calc_res_utilities(self) -> None:
        """
        Calculates residential utility at a household its current position and
        given its parameter values.
        """

        b = self.neighbourhood_mixture
        f = self.optimal_fraction
        M = self.utility_at_max
        x = (1 - b) * self.local_compositions + b * self.neighbourhood_compositions

        calc_comp_utility(Household._household_res_utility, x, M, f)

    def calc_school_utilities(self) -> None:
        """
        Calculates school utilities at a student its current school, given
        distance and its other parameter values.
        """

        alpha = self.alpha
        f = self.optimal_fraction
        M = self.utility_at_max
        x = self.school_compositions

        calc_comp_utility(Household._household_school_utility_comp, x, M, f)

        # TODO: This needs to be correct, what distances to use?
        Household._household_school_utility = (
            Household._household_school_utility_comp * alpha
        ) + (Household._household_distance * (1 - alpha))

    def calc_school_rankings(
        self, households: List[Household], schools: List[School]
    ) -> None:
        """
        Ranks the schools according to utility.
        Note: this is probably a performance bottleneck; one of the array is
        of size [n_households_moving, n_of_schools].
        Args:
            households (list): list of households the rankings need to be
                calculated for.
            schools (list): list of schools that need to be ranked.
        Todo:
            Schools can differ per household if we only want to look at the
            n-closest schools for example?
        """
        if len(schools) == 0 or len(households) == 0:
            return

        zeros = np.zeros(len(self.params["group_types"][0]))
        compositions = np.array(
            [
                school.composition / school.total if school.total > 0 else zeros
                for school in schools
            ],
            dtype="float32",
        )

        households_data = np.array(
            [(h.idx, h.category) for h in households], dtype=int
        ).T
        households_indices = households_data[0, :]
        households_categories = households_data[1, :]

        # Composition utility calculations
        t = np.take(self.optimal_fraction, households_indices)  # [n_indices]
        M = np.take(self.utility_at_max, households_indices)  # [n_indices]
        x = np.take(
            compositions, households_categories, axis=1
        )  # [n_schools, n_indices]
        composition_utilities = np.where(x <= t, x / t, M + (1 - x) * (1 - M) / (1 - t))

        # Combined (THIS SHOULD BE GENERALISED TO INCLUDE MORE FACTORS)
        utilities = (
            composition_utilities * self.alpha[np.newaxis, households_indices]
            + (
                np.take(self.distance_utilities, households_indices, axis=0)
                * (1 - self.alpha[households_indices, np.newaxis])
            ).T
        )
        # utilities.shape = (len(schools), len(households))

        # Rank the schools according to the household utilities
        # TODO: is this the right utility? ie. residential=False?
        households_utilities = np.take(
            Household._household_school_utility, households_indices
        )

        method = self.params["ranking_method"].lower()
        if method == "proportional":
            differences = utilities - households_utilities[np.newaxis, :]
            exp_utilities = np.exp(self.temperature * differences)
            transformed = exp_utilities / exp_utilities.sum(axis=0)[np.newaxis, :]

        elif method == "highest":
            transformed = utilities

        else:
            print("Method not implementd.")
            sys.exit()

        # instead of reversing the list, sort the negative values in the list
        ranked_indices = (-transformed).argsort(axis=0, kind='quicksort')

        # necessary to allow indexing with the argsort result
        schools = np.array(schools)

        for i, household in enumerate(households):
            ranking = schools[ranked_indices[:, i]]

            # TODO: can we make school preference a Household property,
            # instead of a Student property? gives ~10% extra speedup
            for student in household.students:
                student.set_school_preference(ranking)

    def get_attributes(self, pos: tuple[float, float]) -> np.ndarray:

        """
        Returns the attribute vector of a given position
        Args:
            pos (tuple): (x,y) coordinates.
        Returns:
            Numpy array: containing all the attributes (all zeros if empty)
        """
        return self.household_attrs[pos[0], pos[1], :]

    def switch_attrs(
        self, pos1: tuple[float, float], pos2: tuple[float, float]
    ) -> None:
        """
        Switches two attribute vectors in the attribute grid by making a copy.
        Args:
            pos1 (tuple): (x,y) coordinates.
            pos2 (tuple): (x,y) coordinates.
        """
        temp = np.copy(self.household_attrs[pos1])
        self.household_attrs[pos1] = self.household_attrs[pos2]
        self.household_attrs[pos2] = temp

    def step(self, residential: bool = False, initial_schools: bool = False) -> None:
        """
        Perform model steps.
        Args:
            residential (bool): True if a residential step needs to be done,
                False (default) means a school step.
            initial_schools (bool): True if an initial school step needs to be
                done, False (default) means a school step.
        """

        # Perform school or residential step.
        self.scheduler.step(residential=residential, initial_schools=initial_schools)

    def simulate(self, res_steps: int = None, school_steps: int = None) -> None:
        """
        Performs #res_steps of residential steps and #school_steps of school
        steps.
        Args:
            res_steps (int): Number of residential steps.
            school_steps (int): Number of school steps.
            initial_schools (bool): True if an initial school step needs to be
                done, False (default) means a school step.
        """

        if not res_steps:
            res_steps = self.params["max_res_steps"]

        if not school_steps:
            school_steps = self.params["max_school_steps"]

        while self.scheduler.get_time("residential") < res_steps and not self.res_ended:

            if self.verbose:
                f = (
                    "Residential process: step "
                    + str(self.scheduler.get_time("residential") + 1)
                    + " from "
                    + str(res_steps)
                )
                sys.stdout.write("\r" + f)
                sys.stdout.flush()

            self.res_ended = self.convergence_check()
            if not self.res_ended:
                self.step(residential=True)
            else:
                break

        if self.verbose:
            print()

        while (
            self.scheduler.get_time("school") < school_steps and not self.school_ended
        ):

            if self.verbose:
                f = (
                    "School process: step "
                    + str(self.scheduler.get_time("school") + 1)
                    + " from "
                    + str(school_steps)
                )
                sys.stdout.write("\r" + f)
                sys.stdout.flush()

            self.school_ended = self.convergence_check()

            if self.scheduler.school_steps == 0:
                self.step(residential=False, initial_schools=True)
            else:
                if not self.school_ended:
                    self.step(residential=False, initial_schools=False)
                else:
                    break

        if self.verbose:
            print()
            print("Processes ended")
        self.export_data(self.export)

    def convergence_check(self) -> bool:
        """
        Checks if the processes have converged.
        Returns: True if converged.
        """
        window_size = self.params["window_size"]
        time = self.scheduler.get_time()
        school_time = self.scheduler.get_time("school")

        # Check what type of segregation to calculate (i.e., which of the
        # processes is running)
        if not self.res_ended:
            self.segregation.append(
                self.measurements.calculate_segregation(
                    type="bounded_neighbourhood", index="Theil"
                )
            )
        else:
            self.segregation.append(
                self.measurements.calculate_segregation(type="school", index="Theil")
            )

        # Wait until there is enough steps in the school process
        if self.res_ended and school_time < window_size:
            return False

        # Check all metrics in the window size and check if they are below
        # the convergence threshold
        if time >= window_size - 1:
            utilities = self.measurements.households[
                time - window_size + 1 : time + 1, :, 4
            ]
            means = utilities.mean(axis=1)
            stds = utilities.std(axis=1)

            metrics = np.vstack(
                (means, stds, self.segregation[time - window_size + 1 : time + 1])
            )

            metric_means = np.repeat(
                metrics.mean(axis=1)[:, np.newaxis], window_size, axis=1
            )
            mad = np.abs(metrics - metric_means)
            if np.all(mad < self.params["conv_threshold"]):
                # Start over if the residential process has converged
                self.res_ended = True
                return True

        return False

    def choose_locations(
        self, amount: int, method: str = "evenly_spaced"
    ) -> List[tuple[float, float]]:
        """
        Compute a number of locations to place school and neighbourhood objects.
        Args:
            amount (int): the number of agents to place.
            method (str): one of 'evenly_spaced', 'random', 'random_per_neighbourhood'
        Returns:
            list: containing all the locations in tuple (x,y) format.
        """

        if amount == 0:
            return []

        if method == "evenly_spaced":
            per_side = np.sqrt(amount)
            if per_side % 1 != 0:
                print("Unable to place amount of locations using given method")
                sys.exit(1)

            # Compute locations
            per_side = int(per_side)
            xs = np.linspace(0, self.params["width"], per_side * 2 + 1)[1::2]
            ys = np.linspace(0, self.params["height"], per_side * 2 + 1)[1::2]
            return [(x, y) for x in xs for y in ys]

        elif method == "random":
            locations = []
            i = 0
            while i < amount:
                x_coord = self.random.randint(low=0, high=self.params["width"])
                y_coord = self.random.randint(low=0, high=self.params["height"])
                if (x_coord, y_coord) not in locations:
                    locations.append((x_coord, y_coord))
                    i += 1
            return locations

        elif method == "random_per_neighbourhood":
            locations = []
            width, height = self.params["width"], self.params["height"]
            n_schools = self.params["n_schools"]
            n_neighbourhoods = self.params["n_neighbourhoods"]
            per_side = int(np.sqrt(n_neighbourhoods))
            location_width = width / per_side
            location_height = height / per_side

            # Draw a random sample per neighbourhood as long as there are
            # schools to place
            i = 0
            while i < max(n_neighbourhoods, n_schools):
                y_low = 0
                for col in range(per_side):
                    x_low = 0
                    y_high = int((1 + col) * location_height)

                    for row in range(per_side):

                        x_high = int((1 + row) * location_width)

                        if x_high >= width:
                            x_high = width - 1
                        elif y_high >= height:
                            y_high = height - 1

                        x_coord = self.random.randint(low=x_low, high=x_high)
                        y_coord = self.random.randint(low=y_low, high=y_high)

                        # Check if the coordinates haven't already been sampled
                        while (x_coord, y_coord) in locations:
                            x_coord = self.random.randint(low=x_low, high=x_high)
                            y_coord = self.random.randint(low=y_low, high=y_high)

                        locations.append((x_coord, y_coord))
                        x_low = x_high + 1
                        i += 1
                    y_low = y_high + 1

            # Shuffle all locations if n_schools <= n_neighbourhoods, otherwise
            # shuffle only the remainder
            if n_schools <= n_neighbourhoods:
                self.random.shuffle(locations)
            else:
                divider = int(n_schools / n_neighbourhoods)
                remainder = n_schools % n_neighbourhoods
                first_locations = locations[: n_neighbourhoods * divider]
                rest_locations = locations[n_neighbourhoods * divider :]
                self.random.shuffle(rest_locations)
                locations = first_locations + rest_locations[:remainder]

            return locations

        print(f"Unknown method in choose_locations {method} __file__:__line__")
        sys.exit(-1)

    def compute_closest_neighbourhoods(self) -> dict:
        """
        Compute distance from all grid cells to all schools and all
        neighbourhood objects.
        Returns:
            dict: of dicts containing all Euclidean distances
        """

        EPS = 1.2e-6
        closest_neighbourhoods = {}
        neighbourhoods = self.get_agents("neighbourhoods")

        for x in range(self.params["width"]):
            for y in range(self.params["height"]):

                # Loop over all neighbourhoods, calculate distance and save
                # closest
                point = Point(x, y)
                for neighbourhood in neighbourhoods:
                    shape = neighbourhood.shape.buffer(EPS)
                    if shape.contains(point):
                        closest_neighbourhoods[str((x, y))] = str(neighbourhood.pos)
                        break

        return closest_neighbourhoods

    def compute_school_distances(self) -> None:
        """
        Computes school distances.
        """
        self.all_distances = np.zeros(
            (self.params["n_households"], self.params["n_schools"])
        )
        school_frame = gpd.GeoSeries(
            [Point(school.pos) for school in self.get_agents("schools")]
        )
        for household in self.get_agents("households"):
            self.all_distances[household.idx, :] = school_frame.distance(
                Point(household.pos)
            )

    def get_agents(self, agent_type: str) -> List[object]:
        """
        Returns list of agents of given type.
        Args:
            agent_type (str): either 'School', 'Neighbourhood', 'Household' or
            'Student'.
        Returns:
            list: containing all the objects of the specified type.
        """
        return self.agents[agent_type]

    def export_data(self, export=False) -> None:
        """
        Export data for visualization.
        """
        if export:
            self.measurements.export_data()

    def increment_agent_count(self) -> None:
        """
        Increment agent count by one.
        """
        self.agents["amount"] += 1
