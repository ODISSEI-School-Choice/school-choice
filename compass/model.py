"""
The Model class which initialises the system and all of its components.
"""
from datetime import datetime
import os
import sys
import random
import contextlib
import numpy as np
from mesa import Model
import geopandas as gpd
from .utils import Measurements
from scipy.stats import truncnorm
from scipy.ndimage import convolve
from .agents_household import Household
from mesa.space import ContinuousSpace
from shapely.geometry import Point, box
from .scheduler import ThreeStagedActivation
from .agents_spatial import School, Neighbourhood


@contextlib.contextmanager
def record_time(name):
    try:
        start_time = datetime.now()
        yield
    finally:
        print("\n%s: %f" % (name,
                            (datetime.now() - start_time).total_seconds()))


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
        global_composition_normalized (dict): normalized system compositions.
    """

    def __init__(self, params, export=False):

        super().__init__()

        # Initialise the model attributes
        self.set_attributes(params=params, export=export)

        if self.params['case'].lower() != 'lattice':
            self.load_agents(self.params['case'].lower())
        else:
            self.create_agents()

        # Initial compositions need to be calculated after every household is
        # placed
        for household in self.get_agents('households'):
            household.update_utilities()

        # Get values of the initial configuration
        self.measurements.end_step(residential=True)

        # Calculate global compositions for the segregation calculations
        self.global_composition = self.measurements.neighbourhoods[
            0, :, :2].sum(axis=0)
        self.global_composition_normalized = self.global_composition / \
            self.global_composition.sum()

        if self.verbose:
            text = f""" Model initialised:
                NR AGENTS:  Households: {self.params['n_households']}
                Neighbourhoods: {self.params['n_neighbourhoods']}
                Schools: {self.params['n_schools']}
                In scheduler: {self.scheduler.get_agent_count()}"""
            print(text)

    def set_attributes(self, params, export=False):
        """
        Sets or calculates all attributes used in the Compass class.

        Args:
            params (Argparser): containing all parameter values.
            export (bool): True if the data needs to be exported or not.
        """

        # Calculate number of households and students
        params["n_households"] = int(
            params["household_density"] *
            (params["width"] * params["height"] - params["n_neighbourhoods"] -
             params["n_schools"]))
        params['n_students'] = int(params["n_households"] *
                                   params["student_density"])
        self.params = dict(params)

        # Set tracking attributes
        self.export = export
        self.segregation = []  # Track segregation over time
        self.res_ended = False
        self.school_ended = False
        self.verbose = self.params['verbose']
        self.agents = {
            "amount": 0,
            "households": [],
            "schools": [],
            "neighbourhoods": []
        }

        # Initialise other objects
        self.measurements = Measurements(self)
        self.scheduler = ThreeStagedActivation(self)
        self.grid = ContinuousSpace(self.params["width"],
                                    self.params["height"],
                                    torus=self.params["torus"])

    def create_agents(self):
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

    def set_agent_parameters(self, params, households):
        """
        Puts the agent parameters in numpy arrays for faster computations.

        Args:
            params (dict): Model parameters which could differ from
                the agent params!
            households (list): list of Household objects. Students should
                inherit their parameters from the Household object.

        Todo:
            Parameters should be imported from a config file in the future.
        """

        # to remember the index in the array of the specific household
        n_agents = len(households)
        self.local_compositions = []
        self.neighbourhood_compositions = []
        schools = self.get_agents('schools')
        n_schools = len(schools)

        dtype = "float32"
        self.alpha = np.zeros(n_agents, dtype=dtype)
        self.temperature = self.params['temperature']
        self.categories = np.zeros(n_agents, dtype=int)
        self.utility_at_max = np.zeros(n_agents, dtype=dtype)
        self.optimal_fraction = np.zeros(n_agents, dtype=dtype)
        self.neighbourhood_mixture = np.ones(n_agents, dtype=int)

        optimal_fractions = self.trunc_normal_sample(
            params["optimal_fraction"][0],
            scale=params['homophily_std'],
            size=n_agents)
        alphas = self.trunc_normal_sample([params["alpha"], params['alpha']],
                                          scale=params['homophily_std'],
                                          size=n_agents)
        utility_at_maxs = self.trunc_normal_sample(
            params["utility_at_max"][0],
            scale=params['homophily_std'],
            size=n_agents)

        distances = np.zeros((n_agents, n_schools), dtype=dtype)
        school_objects = np.zeros((n_agents, n_schools), dtype=object)

        if self.params['case'].lower() == 'lattice':
            local_compositions = self.normalized_compositions

        array_index = 0

        for household in households:
            household.array_index = array_index
            x, y = household.pos

            # Fill arrays with agent parameter values for faster computations
            self.categories[array_index] = household.category
            self.optimal_fraction[array_index] = optimal_fractions[
                household.category][array_index]
            self.alpha[array_index] = alphas[household.category][array_index]
            self.utility_at_max[array_index] = utility_at_maxs[
                household.category][array_index]

            # Currently only convolution (assumes every household has the same
            # radius) for composition calculations within the lattice case.
            if params['case'].lower() == 'lattice':
                self.local_compositions.append(
                    local_compositions[x, y, household.category])
            else:
                household.params['neighbourhood_mixture'] = 1

            if household.neighbourhood.total > 0:
                norm = 1.0 / household.neighbourhood.total
            else:
                norm = 1.0
            self.neighbourhood_compositions.append(
                household.neighbourhood.composition[
                    household.category] * norm)
            array_index += 1

        self.school_objects = school_objects

        # These are filled with the actual distance and composition utilities
        # of the household and the school (singular!) they attend

        # SHOULD BE CALLED DIFFERENTLY CAUSE NOW IT OVERWRITES AN ATTRIBUTE!!!
        self.distances = np.zeros(n_agents, dtype=dtype)
        self.school_compositions = np.zeros(n_agents, dtype=dtype)

        # Distance utilities based on sigmoid function
        if self.params['case'].lower() != 'lattice':
            p = self.params['p']
            q = self.params['q']
            self.distance_utilities = 1. / (1 + (self.all_distances / p)**q)

        self.vectorise_functions()  # for element-wise operations

    def vectorise_functions(self):
        """
        Vectorises functions using numpy.vectorize for use in
        array computations.
        """
        self.calc_comp_utility_v = np.vectorize(self.calc_comp_utility)

    def calc_comp_utility(self, x, M, f):
        """
        Calculates the utility given a normalised composition (0<=x<=1), an
        optimal fraction (0<=f<=1) and utility at homogeneity (0<=M<=1).
        """
        if x <= f:
            utility = x / f
        else:
            utility = M + (1 - x) * (1 - M) / (1 - f)
        return utility

    def neighbourhoods(self):
        """
        Adds the neighbourhood objects to the environment.
        """

        n_neighs = self.params["n_neighbourhoods"]

        # Add neighbourhoods if necessary
        if n_neighs:
            locations = self.choose_locations(
                n_neighs, self.params["neighbourhoods_placement"])
            for i in range(n_neighs):

                x, y = locations[i]
                location = (x, y)
                size = self.params['width'] / float(n_neighs**0.5 * 2)
                minx, miny = x - size, y - size
                maxx, maxy = x + size, y + size
                shape = box(minx, miny, maxx, maxy)

                # Create the Neighbourhood object and place it on the grid and
                # add it to the scheduler
                neighbourhood = Neighbourhood(self.get_agents("amount"),
                                              location, shape, self,
                                              self.params)
                self.agents["neighbourhoods"].append(neighbourhood)
                self.scheduler.add(neighbourhood)
                self.grid.place_agent(neighbourhood, locations[i])

    def schools(self):
        """
        Adds the school objects to the environment.
        """

        # Add schools if necessary
        if self.params["n_schools"]:
            locations = self.choose_locations(self.params["n_schools"],
                                              self.params["schools_placement"])
            for i in range(self.params["n_schools"]):
                x, y = locations[i]
                location = (x, y)

                # Create the School object and place it on the grid and add it
                # to the scheduler
                school = School(self.get_agents("amount"), location, self,
                                self.params)
                school.array_index = i
                self.agents["schools"].append(school)
                self.scheduler.add(school)
                self.grid.place_agent(school, location)

    def households(self):
        """
        Adds the household objects to the environment.

        Todo:
            * Place household specific parameters in a attribute called params
        """

        params = self.params
        self.chosen_indices = None  # Only matters for case studies
        self.household_attrs = np.zeros(shape=(params["width"],
                                               params['height'],
                                               len(params['group_types'][0])),
                                        dtype="float32")

        # Create group types, empty spots and shuffle them both
        n_groups = len(params["group_categories"])
        groups = [
            np.random.choice(list(range(0, len(params["group_types"][i]))),
                             size=params["n_households"],
                             p=params["group_dist"][i])
            for i in range(n_groups)
        ]

        self.grid.empties = set([(x, y) for x in range(params['width'])
                                 for y in range(params['height'])])
        empties = list(self.grid.empties)
        np.random.shuffle(empties)

        for i, position in enumerate(empties[0:params["n_households"]]):
            household = Household(self.get_agents("amount"), position, self,
                                  params, groups[0][i])

            # Place households on the grid and add them to the scheduler
            self.agents["households"].append(household)
            self.grid.place_agent(household, position)
            self.grid.empties.discard(position)
            self.scheduler.add(household)
            self.household_attrs[position[0],
                                 position[1], :] = household.attributes

        # Calculate AFTER all agents are placed
        all_households = self.get_agents('households')
        self.calc_residential_compositions()
        self.set_agent_parameters(params, all_households)
        self.calc_res_utilities()
        # (household.update_utilities() for household in all_households)

    def load_agents(self, case='Amsterdam'):
        """
        Load the agents from several files.

        Note:
            This function is in progress and works only for the
            Amsterdam case now.
        """

        dirname = os.path.dirname(__file__)
        if case.lower() == 'amsterdam':
            path = dirname + '/maps/amsterdam'

        # Load GeoDataFrames
        school_frame = gpd.read_file(path + '/schools.geojson')
        household_frame = gpd.read_file(path + '/households.geojson')
        neighbourhood_frame = gpd.read_file(path + '/neighbourhoods.geojson')

        # Create grid
        self.params["torus"] = 0
        self.params['max_res_steps'] = 0
        xmin, ymin, xmax, ymax = neighbourhood_frame.total_bounds
        self.grid = ContinuousSpace(xmax, ymax, self.params["torus"], xmin,
                                    ymin)
        self.grid.empties = [(0, 0)]

        # In the file more households could be available to sample from,
        # but only use the actual amount
        data = np.load(path + '/distances_perc_of_actual.npz')
        perc_of_actual = data['perc_of_actual']
        self.all_distances = data['distances']

        self.scheduler = ThreeStagedActivation(self)

        # More agents are simulated to sample from them and incorporate some
        # randomness in the type and spatial distribution
        total_households = len(household_frame)
        actual_households = int(total_households / perc_of_actual)
        self.params['n_households'] = actual_households
        self.params['n_students'] = int(self.params["n_households"] *
                                        self.params["student_density"])

        # Create neighbourhoods
        self.create_neighbourhoods(neighbourhood_frame)

        # Create schools
        self.create_schools(school_frame)

        # Create households
        self.create_households(household_frame, actual_households)

        if self.verbose:
            print('Setting agent parameters...')
        self.set_agent_parameters(self.params, self.agents["households"])

        self.local_compositions = self.neighbourhood_compositions
        self.calc_res_utilities()

        if self.verbose:
            print('Model loaded!')

    def create_neighbourhoods(self, neighbourhood_frame):
        """
        Given a GeoDataFrame, this creates all the neighbourhood objects
        """
        if self.verbose:
            print("Creating neighbourhoods...")

        self.params['n_neighbourhoods'] = len(neighbourhood_frame)
        for index, row in neighbourhood_frame.iterrows():
            neighbourhood = Neighbourhood(unique_id=index,
                                          pos=(row.geometry.centroid.xy[0][0],
                                               row.geometry.centroid.xy[1][0]),
                                          shape=row.geometry,
                                          model=self,
                                          params=self.params)
            self.agents["neighbourhoods"].append(neighbourhood)
            self.scheduler.add(neighbourhood)
            self.grid.place_agent(neighbourhood, neighbourhood.pos)

    def create_schools(self, school_frame):
        """
        Given a GeoDataFrame, this creates all the school objects
        """
        if self.verbose:
            print("Creating schools...")

        self.params['n_schools'] = len(school_frame)
        n_neighbourhoods = self.params['n_neighbourhoods']
        for index, row in school_frame.iterrows():
            school = School(unique_id=index + n_neighbourhoods,
                            pos=(row.geometry.xy[0][0], row.geometry.xy[1][0]),
                            model=self,
                            params=self.params)
            school.array_index = index
            school.capacity = 1 + int(self.params["school_capacity"] * \
                        self.params["n_students"] / self.params["n_schools"])
            self.agents["schools"].append(school)
            self.scheduler.add(school)
            self.grid.place_agent(school, school.pos)

    def create_households(self, household_frame, actual_households):
        """
        Given a GeoDataFrame, this creates all the household objects
        """
        if self.verbose:
            print("Creating households...")

        self.chosen_indices = np.random.choice(len(household_frame),
                                               size=actual_households,
                                               replace=False)
        households = household_frame.iloc[self.chosen_indices]

        if self.params['random_residential']:
            # Randomly shuffle the group of the household
            shuffled = households['group'].values
            np.random.shuffle(shuffled)
            households['group'] = shuffled

        self.params['n_households'] = len(households)
        n_agents = self.params['n_neighbourhoods'] + self.params['n_schools']
        neighbourhoods = self.get_agents('neighbourhoods')

        self.all_distances = self.all_distances[self.chosen_indices, :]

        for index, row in households.iterrows():
            self.create_household(index, row, n_agents, neighbourhoods)

        self.location_to_agent()

    def create_household(self, index, row, n_agents, neighbourhoods):
        """
        Creates ONE household
        """
        household = Household(unique_id=index + n_agents,
                              pos=row.geometry.coords[0],
                              model=self,
                              params=self.params,
                              category=row['group'],
                              nhood=neighbourhoods[row['neighbourhood_id']])
        household.array_index = index
        self.agents["households"].append(household)
        self.scheduler.add(household)
        self.grid.place_agent(household, household.pos)

    def location_to_agent(self):
        """
        Creates a dictionary with the location of the neighbourhoods as key and
        the object itself as value. Schools are not included as they can have
        the same position as a neighbourhood (centroid).
        """
        agents = self.get_agents('neighbourhoods')
        self.location_to_agent = {str(agent.pos): agent for agent in agents}

    def trunc_normal_sample(self, means, scale, size):
        """
        Samples from a truncated normal distribution.
        """
        sample = [0] * len(means)
        for index, mu in enumerate(means):

            # All samples are equal if the scale is zero.
            if scale == 0:
                sample[index] = np.repeat(mu, size)
            else:
                sample[index] = truncnorm.rvs((0 - mu) / scale,
                                              (1 - mu) / scale,
                                              loc=mu,
                                              scale=scale,
                                              size=size)
        return sample

    def calc_residential_compositions(self):
        """
        Updates all local residential compositions assuming all households have
        the SAME RADIUS.
        """

        # Determine the kernel of the convolution
        radius = self.params['radius']
        dim = radius * 2 + 1
        self.kernel = np.ones((dim, dim))
        self.kernel[radius, radius] = 0

        # Should it wrap around the edges or not?
        if self.params['torus']:
            mode = 'wrap'
        else:
            mode = 'constant'

        summed = 0
        num_attrs = self.household_attrs.shape[2]
        compositions = np.zeros(shape=self.household_attrs.shape,
                                dtype="float32")

        # Convolution for every household attribute.
        for attr in range(num_attrs):
            compositions[:, :, attr] = convolve(self.household_attrs[:, :,
                                                                     attr],
                                                self.kernel,
                                                mode=mode)
            summed += compositions[:, :, attr]
        self.compositions = compositions
        self.normalized_compositions = np.nan_to_num(
            compositions /
            np.repeat(summed[:, :, np.newaxis], num_attrs, axis=2))

    def calc_school_compositions(self):
        """
        Calculate the new school compositions for every household and only for
        the first student!

        Note:
            Currently only for the first student!!!
        """

        for household in self.get_agents('households'):
            category = household.category
            array_index = household.array_index
            self.distances[array_index] = household.distance
            if household.students[0].school.total > 0:
                norm = 1.0 / household.students[0].school.total
            else:
                norm = 1.0
            self.school_compositions[array_index] = \
                household.students[0].school.composition[category] * norm

    def calc_res_utilities(self):
        """
        Calculates residential utility at a household its current position and
        given its parameter values.
        """

        b = self.neighbourhood_mixture
        f = self.optimal_fraction
        M = self.utility_at_max
        x = (1-b)*self.local_compositions + \
            b*self.neighbourhood_compositions
        self.res_utilities = self.calc_comp_utility_v(x, M, f)

    def calc_school_utilities(self):
        """
        Calculates school utilities at a student its current school, given
        distance and its other parameter values.
        """

        alpha = self.alpha
        f = self.optimal_fraction
        M = self.utility_at_max
        x = self.school_compositions
        self.school_composition_utilities = self.calc_comp_utility_v(x, M, f)

        # TODO: This needs to be correct, what distances to use?
        # self.school_utilities = (self.school_composition_utilities ** alpha) * \
        #     (self.distances ** (1 - alpha))

        self.school_utilities = (self.school_composition_utilities * alpha) + \
            (self.distances * (1 - alpha))

    def calc_school_rankings(self, households, schools):
        """
        Ranks the schools according to utility.

        Args:
            households (list): list of households the rankings need to be
                calculated for.
            schools (list): list of schools that need to be ranked.

        Todo:
            Schools can differ per household if we only want to look at the
            n-closest schools for example?
        """
        zeros = np.zeros(len(self.params["group_types"][0]))
        compositions = np.array(
            [school.composition / school.total if school.total > 0 else zeros for school in schools],
            dtype="float32")

        # Composition utility calculations
        t = self.optimal_fraction
        M = self.utility_at_max
        x = compositions[:, self.categories]
        composition_utilities = np.where(x <= t, x / t,
                                         M + (1 - x) * (1 - M) / (1 - t))

        # Combined (THIS SHOULD BE GENERALISED TO INCLUDE MORE FACTORS)
        utilities = composition_utilities * self.alpha[np.newaxis, :] + \
            (self.distance_utilities * (1 - self.alpha[: ,np.newaxis])).T

        # Rank the schools according to the household utilities
        schools = np.array(schools)
        if self.params['ranking_method'].lower() == 'proportional':
            transform = True
        else:
            transform = False

        # for household in households:
        #     if transform:
        #         # Some randomness according to the temperature parameter
        #         differences = utilities[:, household.array_index] - household.utility
        #         exp_utilities = np.exp(self.temperature*differences)
        #         transformed = exp_utilities / exp_utilities.sum()
        #     else:
        #         # Pick highest utility
        #         transformed = utilities[:, household.array_index]

        #     ranked_idx = transformed.argsort()[::-1]
        #     ranking = schools[ranked_idx]
        #     [student.set_school_preference(ranking) for student in household.students]

        # vectorization of the code above
        households_indices = [h.array_index for h in households]
        households_utilities = np.fromiter([h.utility for h in households],
                                           dtype="float32")
        transformed = utilities[:, households_indices]
        if transform:
            differences = transformed - households_utilities[np.newaxis, :]
            exp_utilities = np.exp(self.temperature * differences)
            transformed = exp_utilities / exp_utilities.sum(
                axis=0)[np.newaxis, :]
        ranked_indices = transformed.argsort(axis=0)[::-1]

        for i in range(len(households)):
            ranking = schools[ranked_indices[:, i]]
            for s in households[i].students:
                s.set_school_preference(ranking)

    def get_attributes(self, pos):
        """
        Returns the attribute vector of a given position

        Args:
            pos (tuple): (x,y) coordinates.

        Returns:
            Numpy array: containing all the attributes (all zeros if empty)
        """
        return self.household_attrs[pos[0], pos[1], :]

    def switch_attrs(self, pos1, pos2):
        """
        Switches two attribute vectors in the attribute grid by making a copy.

        Args:
            pos1 (tuple): (x,y) coordinates.
            pos2 (tuple): (x,y) coordinates.
        """
        temp = np.copy(self.household_attrs[pos1])
        self.household_attrs[pos1] = self.household_attrs[pos2]
        self.household_attrs[pos2] = temp

    def step(self, residential=False, initial_schools=False):
        """
        Perform model steps.

        Args:
            residential (bool): True if a residential step needs to be done,
                False (default) means a school step.
            initial_schools (bool): True if an initial school step needs to be
                done, False (default) means a school step.
        """

        # Perform school or residential step.
        self.scheduler.step(residential=residential,
                            initial_schools=initial_schools)

    def simulate(self, res_steps=None, school_steps=None):
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
            res_steps = self.params['max_res_steps']

        if not school_steps:
            school_steps = self.params['max_school_steps']

        while (self.scheduler.get_time('residential') < res_steps \
            and not self.res_ended):

            if self.verbose:
                f = "Residential process: step " + str(
                    self.scheduler.get_time('residential')+1) + " from " + \
                    str(res_steps)
                sys.stdout.write("\r" + f)
                sys.stdout.flush()

            self.res_ended = self.convergence_check()
            if not self.res_ended:
                self.step(residential=True)
            else:
                break

        if self.verbose:
            print()

        while (self.scheduler.get_time('school') < school_steps \
            and not self.school_ended):

            if self.verbose:
                f = "School process: step " + str(
                    self.scheduler.get_time('school')+1) + " from " + \
                    str(school_steps)
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

    def convergence_check(self):
        """
        Checks if the processes have converged.

        Returns: True if converged.
        """
        window_size = self.params['window_size']
        time = self.scheduler.get_time()
        school_time = self.scheduler.get_time('school')

        # Check what type of segregation to calculate (i.e., which of the
        # processes is running)
        if not self.res_ended:
            self.segregation.append(
                self.measurements.calculate_segregation(
                    type="bounded_neighbourhood", index="Theil"))
        else:
            self.segregation.append(
                self.measurements.calculate_segregation(type="school",
                                                        index="Theil"))

        # Wait until there is enough steps in the school process
        if (self.res_ended and school_time < window_size):
            return False

        # Check all metrics in the window size and check if they are below
        # the convergence threshold
        if time >= window_size - 1:
            utilities = self.measurements.households[time - window_size +
                                                     1:time + 1, :, 4]
            means = utilities.mean(axis=1)
            stds = utilities.std(axis=1)

            metrics = np.vstack(
                (means, stds,
                 self.segregation[time - window_size + 1:time + 1]))

            metric_means = np.repeat(metrics.mean(axis=1)[:, np.newaxis],
                                     window_size,
                                     axis=1)
            mad = np.abs(metrics - metric_means)
            if np.all(mad < self.params["conv_threshold"]):
                # Start over if the residential process has converged
                self.res_ended = True
                return True

        return False

    def choose_locations(self, amount, method="evenly_spaced"):
        """
        Compute a number of locations to place school and neighbourhood objects.
        Currently, only random and evenly spaced locations are allowed.

        Args:
            amount (int): the number of agents to place.
            method (str): 'evenly_spaced' only supported method for now.

        Returns:
            list: containing all the locations in tuple (x,y) format.
        """

        if amount == 0:
            return []

        # Evenly spaced is also used in 'random_per_neighbourhood'
        per_side = np.sqrt(amount)
        if per_side % 1 != 0:
            print("Unable to place amount of locations using given method")
            sys.exit(1)

        # Compute locations
        locations = []

        if method == "random":
            i = 0
            while i < amount:
                x_coord = np.random.randint(low=0, high=self.params['width'])
                y_coord = np.random.randint(low=0, high=self.params['height'])
                if (x_coord, y_coord) not in locations:
                    locations.append((x_coord, y_coord))
                    i += 1

        elif method == "random_per_neighbourhood":

            width, height = self.params["width"], self.params['height']
            n_schools = self.params['n_schools']
            n_neighbourhoods = self.params['n_neighbourhoods']
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

                        x_coord = np.random.randint(low=x_low, high=x_high)
                        y_coord = np.random.randint(low=y_low, high=y_high)

                        # Check if the coordinates haven't already been sampled
                        while (x_coord, y_coord) in locations:
                            x_coord = np.random.randint(low=x_low, high=x_high)
                            y_coord = np.random.randint(low=y_low, high=y_high)

                        locations.append((x_coord, y_coord))
                        x_low = x_high + 1
                        i += 1
                    y_low = y_high + 1

            # Shuffle all locations if n_schools <= n_neighbourhoods, otherwise
            # shuffle only the remainder
            if n_schools <= n_neighbourhoods:
                random.shuffle(locations)
            else:
                divider = int(n_schools / n_neighbourhoods)
                remainder = n_schools % n_neighbourhoods
                first_locations = locations[:n_neighbourhoods * divider]
                rest_locations = locations[n_neighbourhoods * divider:]
                random.shuffle(rest_locations)
                locations = first_locations + rest_locations[:remainder]

        return locations

    def compute_closest_neighbourhoods(self):
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
                        closest_neighbourhoods[str(
                            (x, y))] = str(neighbourhood.pos)
                        break

        return closest_neighbourhoods

    def compute_school_distances(self):
        """
        Computes school distances.
        """
        self.all_distances = np.zeros(
            (self.params["n_households"], self.params['n_schools']))
        school_frame = gpd.GeoSeries(
            [Point(school.pos) for school in self.get_agents('schools')])
        for household in self.get_agents('households'):
            self.all_distances[
                household.array_index, :] = school_frame.distance(
                    household.shape)

    def get_agents(self, agent_type):
        """
        Returns list of agents of given type.

        Args:
            agent_type (str): either 'School', 'Neighbourhood', 'Household' or
            'Student'.

        Returns:
            list: containing all the objects of the specified type.
        """
        return self.agents[agent_type]

    def export_data(self, export=False):
        """
        Export data for visualization.
        """
        if export:
            self.measurements.export_data()

    def increment_agent_count(self):
        """
        Increment agent count by one.
        """
        self.agents["amount"] += 1
