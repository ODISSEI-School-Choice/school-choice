"""
The Model class which initialises the system and all of its components.
"""

import os
import sys
import random
import pickle
import logging
import numpy as np
from mesa import Model
from utils import Measurements
from scipy.stats import truncnorm
from scipy.ndimage import convolve
from agents_household import Household
from mesa.space import ContinuousSpace
from shapely.geometry import Point, box
from scheduler import ThreeStagedActivation
from agents_spatial import School, Neighbourhood


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

        # Initialise agents
        if self.logging_enabled:
            logging.debug('Initialising agents')
        if self.params['case'].lower() != 'lattice':
            self.load_agents(self.params['case'].lower())
        else:
            self.create_agents()

        # Get values of the initial configuration
        self.measurements.end_step(residential=True)

        # Calculate global compositions for the segregation calculations
        self.global_composition = self.measurements.neighbourhoods[
            0, :, :2].sum(axis=0)
        self.global_composition_normalized = self.global_composition / \
            self.global_composition.sum()

        text = f""" Model initialised:
            NR AGENTS:  Households: {self.params['n_households']}
            Neighbourhoods: {self.params['n_neighbourhoods']}
            Schools: {self.params['n_schools']}
            In scheduler: {self.scheduler.get_agent_count()}"""
        print(text)
        if self.logging_enabled:
            logging.debug(text)


    def set_attributes(self, params, export=False):
        """
        Sets or calculates all attributes used in the Compass class.

        Args:
            params (Argparser): containing all parameter values.
            export (bool): True if the data needs to be exported or not.
        """

        # Calculate number of households and students
        
        params["n_households"] = int(params["household_density"]
            * (params["width"] * params["height"]
            - params["n_neighbourhoods"] - params["n_schools"]))
        params['n_students'] = int(params["n_households"] * 
                params["student_density"])
        self.params = dict(params)

        # Set up logger
        self.logging_enabled = params['logging_enabled']
        if self.logging_enabled:     
            self.logger(log_path='logs/', filename='model.log')
            logging.debug('Starting new model')

        # Precalculate shocks
        self.uniform_shocks = np.random.random(1000)
        self.shocks = np.random.normal(0, 0.05, 1000)

        # Set tracking attributes
        self.export = export
        self.segregation = []       # Track segregation over time
        self.res_ended = False
        self.school_ended = False
        self.composition = self.composition_normalized = None
        self.agents = {"amount": 0, "households": [], "schools": [],
            "neighbourhoods": []}
        
        # Initialise other objects
        self.measurements = Measurements(self)
        self.scheduler = ThreeStagedActivation(self)
        self.grid = ContinuousSpace(self.params["width"], self.params["height"],
                                torus=self.params["torus"])


    def create_agents(self):
        """
        Creates the agents when no case study is provided.
        """
        self.neighbourhoods()
        self.schools()
        self.location_to_agent()

        # Compute distance matrix 
        self.distances, \
        self.closest_schools, \
        self.furthest_schools, \
        self.closest_neighbourhoods = self.compute_distances()

        # Compute normalised distances (nearness matrix)
        self.norm_distances = self.compute_norm_distances(
            self.distances, 
            self.closest_schools, 
            self.furthest_schools,
            self.closest_neighbourhoods)

        if self.logging_enabled:
            logging.debug('Placing households and creating attribute matrix')
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
        array_index = 0

        n_agents = len(households)
        self.local_compositions = []
        self.neighbourhood_compositions = []

        self.utility_at_max = np.repeat(params['utility_at_max'][0][0], 
            n_agents)
        self.neighbourhood_mixture = np.repeat(params['neighbourhood_mixture'], 
            n_agents)
        self.alpha = np.repeat(params['alpha'], n_agents)
        self.optimal_fraction = np.repeat(params['optimal_fraction'][0][0], 
            n_agents)

        if self.params['case'].lower() == 'lattice':
            local_compositions = self.normalized_compositions
        
        for household in households:
            household.array_index = array_index
            x,y = household.pos

            # Currently only convolution (assumes every household has the same
            # radius) for composition calculations within the lattice case.
            if self.params['case'].lower() == 'lattice':
                self.local_compositions.append(
                    local_compositions[x,y,household.category])

            self.neighbourhood_compositions.append(
                household.neighbourhood.composition_normalized[household.category])
            array_index += 1

        self.agent_distances = np.zeros(n_agents)
        self.school_compositions = np.zeros(n_agents)
        self.vectorise_functions() # for element-wise operations


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
            utility =  x / f
        else:
            utility =  M + (1-x)*(1-M) / (1-f)
        return utility


    def logger(self, log_path, filename):
        """
        Sets up the logger.
        """

        self.log_path = log_path
        # Check if folder exist, otherwise create it.
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        fileh = logging.FileHandler(log_path + filename, 'a')
        formatter = logging.Formatter('%(asctime)s - %(name)s - ' + \
            '%(levelname)s - %(message)s')
        fileh.setFormatter(formatter)

        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            if isinstance(hdlr, logging.FileHandler):
                log.removeHandler(hdlr)
        log.setLevel(self.params['loglevel'])
        log.addHandler(fileh)      # set the new handler


    def neighbourhoods(self):
        """
        Adds the neighbourhood objects to the environment.
        """

        n_neighs = self.params["n_neighbourhoods"]

        # Add neighbourhoods if necessary
        if n_neighs:
            locations = self.choose_locations(n_neighs,
                                self.params["neighbourhoods_placement"])
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
                self.get_agents("neighbourhoods").append(neighbourhood)
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
                school = School(self.get_agents("amount"),
                    location, self, self.params)
                self.get_agents("schools").append(school)
                self.scheduler.add(school)
                self.grid.place_agent(school, location)


    def households(self):
        """
        Adds the household objects to the environment.

        Todo:
            * Place household specific parameters in a attribute called params
        """

        params = self.params
        self.household_attrs = np.zeros(shape=(params["width"],
            params['height'], len(params['group_types'][0])))

        # Create group types, empty spots and shuffle them both
        n_groups = len(params["group_categories"])
        groups = [np.random.choice(list(range(0,
            len(params["group_types"][i]))), size=params["n_households"],
            p=params["group_dist"][i]) for i in range(n_groups)]

        self.grid.empties = set(
            [(x,y) for x in range(params['width']) 
            for y in range(params['height'])])
        empties = list(self.grid.empties)
        np.random.shuffle(empties)

        # Sample tolerances
        tolerances0, tolerances1 = self.sample_tolerances(
            params["optimal_fraction"][0],
            scale=params['homophily_std'],
            size=params["n_households"])


        for i, position in enumerate(empties[0:params["n_households"]]):
            household = Household(self.get_agents("amount"), position, self,
                params, groups[0][i])

            # Find according group and group specific information
            for j in range(len(groups)):
                category = groups[j][i]
                if category == 0:
                    tolerance = tolerances0[i]
                elif category == 1:
                    tolerance = tolerances1[i]

            # Place households on the grid and add them to the scheduler
            self.get_agents("households").append(household)
            self.grid.place_agent(household, position)
            self.grid.empties.discard(position)
            self.scheduler.add(household)
            self.household_attrs[position[0], position[1], :] = household.attributes

        # Calculate AFTER all agents are placed
        all_households = self.get_agents('households')
        self.calc_residential_compositions()
        self.set_agent_parameters(params, all_households)
        self.calc_res_utilities()
        [household.update_utilities() for household in all_households]


    def load_agents(self, case='Amsterdam'):
        """
        Load the agents from a pickle.

        Note:
            This function is in progress and works only for the Amsterdam and
            IJburg case currently. In the future agents should be able to be 
            imported in a general manner.
        """
        
        
        dirname = os.path.dirname(__file__)
        
        if case.lower()=='ijburg':
            filename = os.path.join(dirname, 'maps/ijburg/agents_ijburg.pickle')
            file = open(filename, 'rb')
        elif case.lower()=='amsterdam':
            filename = os.path.join(dirname, 'maps/amsterdam/agents_3p.pickle')
            file = open(filename, 'rb')
        data = pickle.load(file)

        # Create grid
        self.params["torus"] = 0
        xmin, ymin, xmax, ymax = data['neighbourhoods_gpd'].total_bounds
        self.grid = ContinuousSpace(xmax, ymax, self.params["torus"],
            xmin, ymin)
        self.params['max_res_steps'] = 0
        self.grid.empties = [(0,0)]
        self.closest_neighbourhoods = {}

        agents = data['agents']

        # Create neighbourhoods
        neighbourhoods = agents[agents.object=='Neighbourhood']
        self.params['n_neighbourhoods'] = len(neighbourhoods)
        for index, agent in neighbourhoods.iterrows():
    
            neighbourhood = Neighbourhood(index, (agent.x, agent.y),
                                agent.geometry, self, self.params)
            self.get_agents("neighbourhoods").append(neighbourhood)
            self.scheduler.add(neighbourhood)
            pos = neighbourhood.shape.centroid
            self.grid.place_agent(neighbourhood, (pos.x, pos.y))

        # Create schools
        schools = agents[agents.object=='School']
        self.params['n_schools'] = len(schools)
        for index, agent in schools.iterrows():
            school = School(index, (agent.x, agent.y), self, self.params)
            self.get_agents("schools").append(school)
            self.scheduler.add(school)
            self.grid.place_agent(school, (agent.x, agent.y))

        # Create households
        households = agents[agents.object=='Household']

        # In the file more households could be available to sample from,
        # but only use the actual amount
        try:
            perc_of_actual = data['perc_of_actual']
        except KeyError:
            perc_of_actual = 1
        
        # More agents are simulated to sample from them and incorporate some 
        # randomness in the type and spatial distribution
        total_households = len(households)
        actual_households = int(total_households / perc_of_actual)
        self.params['n_households'] = actual_households
        self.params['n_students'] = int(self.params["n_households"] * 
                self.params["student_density"])

        if self.params['random_residential']:
            # Randomly shuffle the coordinates of the households 
            # not schools and neighbourhoods
            households[['x', 'y']] = households[['x', 'y']].sample(frac=1).values

        households = households.sample(actual_households) # to make sure it's different every time
        
        for index, agent in households.iterrows():
            pos = (agent.x, agent.y)
            for nhood in self.get_agents('neighbourhoods'):
                if nhood.shape.contains(Point(pos)):
                    self.closest_neighbourhoods[str(pos)] = nhood
                    
                    household = Household(index, (agent.x, agent.y), 
                                            self, self.params,
                                            category=int(agent.group),
                                            nhood=nhood)
                    self.get_agents("households").append(household)
                    self.scheduler.add(household)
                    self.grid.place_agent(household, (agent.x, agent.y))
                    break
             
        self.location_to_agent()
        households = self.get_agents('households')

        self.distances = data['distances']
        self.closest_schools = data['closest_schools']
        self.furthest_schools = data['furthest_schools']
        self.norm_distances = self.compute_norm_distances(
            self.distances, 
            self.closest_schools, 
            self.furthest_schools, {})

        self.set_agent_parameters(self.params, households)

        self.local_compositions = self.neighbourhood_compositions
        self.calc_res_utilities()

        # Initial compositions need to be calculated
        for household in households:
            # Only bounded calculation
            household.params['neighbourhood_mixture'] = 1
            household.update_utilities()


    def location_to_agent(self):
        """
        Creates a dictionary with the location of the neighbourhoods as key and
        the object itself as value. Schools are not included as they can have 
        the same position as a neighbourhood (centroid).
        """

        agents = self.get_agents('neighbourhoods')
        self.location_to_agent = {str(agent.pos):agent for agent in agents}


    def sample_tolerances(self, means, scale, size):
        """
        Sample tolerance levels from a truncated normal distribution.
        """
        tolerances = [0]*len(means)
        for index, mu in enumerate(means):

            # All tolerances are equal if the scale is zero.
            if scale==0:
                tolerances[index] = np.repeat(mu, size)
            else:
                tolerances[index] = truncnorm.rvs((0-mu)/scale, (1-mu)/scale,
                                                loc=mu, scale=scale, size=size)
        return tolerances


    def calc_residential_compositions(self):
        """
        Updates all local residential compositions assuming all households have
        the SAME RADIUS.
        """

        # Determine the kernel of the convolution
        radius = self.params['radius']
        dim = radius*2 + 1
        self.kernel = np.ones((dim, dim))
        self.kernel[radius, radius] = 0

        # Should it wrap around the edges or not?
        if self.params['torus']:
            mode='wrap'
        else:
            mode='constant'

        summed = 0
        num_attrs = self.household_attrs.shape[2]
        compositions = np.zeros(shape=self.household_attrs.shape)

        # Convolution for every household attribute.
        for attr in range(num_attrs):
            compositions[:,:,attr] = convolve(
                self.household_attrs[:,:,attr],
                self.kernel, mode=mode)
            summed += compositions[:,:,attr]
        self.compositions = compositions
        self.normalized_compositions = np.nan_to_num(
            compositions / np.repeat(summed[:, :, np.newaxis],
                num_attrs, axis=2))


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
            self.agent_distances[array_index] = household.distance
            self.school_compositions[array_index] = \
                household.students[0].school.composition_normalized[category]
            

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
        distances = self.agent_distances
        self.school_composition_utilities = self.calc_comp_utility_v(x, M, f)
        self.school_utilities = (self.school_composition_utilities ** alpha) * \
            (distances ** (1 - alpha))


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
        
        compositions = np.array(
            [school.composition_normalized for school in schools])

        for household in households:
            
            # Get the normalised distance to every school
            category = household.category
            nearness = self.get_norm_distances(household.pos)
            distances = np.array([nearness[str(school.pos)] for school in schools])

            # Composition utility calculations
            f = household.params['optimal_fraction'][0][category]
            M = household.params['utility_at_max'][0][category]
            x = compositions[:, category]
            composition_utilities = self.calc_comp_utility_v(x, M, f)

            # Combined
            alpha = household.params['alpha']
            utilities = (composition_utilities ** alpha) * \
                (distances ** (1 - alpha))
            utility_dict = dict(zip(schools, utilities))
            ranking = sorted(utility_dict, key=utility_dict.get, reverse=True)
            
            # Set the same ranking for every student
            [student.set_school_preference(ranking) for student in household.students]


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
        message = f'Perform model step, residential = {residential}, initial schools = {initial_schools}'
        if self.logging_enabled:
            logging.debug(message)
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

        logging_level = logging.root.level
        while (self.scheduler.get_time('residential') < res_steps \
            and not self.res_ended):

            if logging_level >= 10:
                f = "Residential process: step " + str(
                self.scheduler.get_time('residential')+1) + " from " + \
                    str(res_steps)
                sys.stdout.write("\r" + f)
                sys.stdout.flush()
                if self.logging_enabled:
                    logging.debug(f)

            self.res_ended = self.convergence_check()
            if not self.res_ended:
                self.step(residential=True)
            else:
                if logging_level >= 10:
                    if self.logging_enabled:
                        logging.debug("\nResidential process converged")
                break
        
        if self.logging_enabled:
            logging.debug("Residential process ended.")
        print()
        while (self.scheduler.get_time('school') < school_steps \
            and not self.school_ended):

            if logging_level >= 10:
                f = "School process: step " + str(
                self.scheduler.get_time('school')+1) + " from " + \
                    str(school_steps)
                sys.stdout.write("\r" + f)
                sys.stdout.flush()
                if self.logging_enabled:
                    logging.debug(f)
            
            self.school_ended = self.convergence_check()
            
            if self.scheduler.school_steps == 0:
                self.step(residential=False, initial_schools=True)
            else:
                if not self.school_ended:
                    self.step(residential=False, initial_schools=False)
                else:
                    if logging_level >= 10:
                        if self.logging_enabled:
                            logging.debug("School process converged")
                    break

        if self.logging_enabled:
            logging.debug("School process ended.")
        if self.logging_enabled:
            logging.debug('Export data')
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
            self.segregation.append(self.measurements.calculate_segregation(
                type="bounded_neighbourhood", index="Theil"))
        else:
            self.segregation.append(self.measurements.calculate_segregation(
                type="school", index="Theil"))
           
        # Wait until there is enough steps in the school process
        if (self.res_ended and school_time<window_size):
            return False

        # Check all metrics in the window size and check if they are below
        # the convergence threshold
        if time >= window_size-1:
            utilities = self.measurements.households[
                time-window_size+1:time+1, :, 4]
            means = utilities.mean(axis=1)
            stds = utilities.std(axis=1)

            metrics = np.vstack((means, stds,
                self.segregation[time-window_size+1:time+1]))
            
            metric_means = np.repeat(metrics.mean(axis=1)[:, np.newaxis],
                window_size, axis=1)
            mad = np.abs(metrics - metric_means)
            if np.all(mad < self.params["conv_threshold"]):
                # Start over if the residential process has converged
                self.res_ended=True
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

        else:
            
            # Evenly spaced is also used in 'random_per_neighbourhood'
            per_side = np.sqrt(amount)
            if per_side % 1 != 0:
                print("Unable to place amount of locations using given method")
                sys.exit(1)

            # Compute locations
            per_side = int(per_side)
            xs = np.linspace(0, self.params['width'], per_side*2+1)[1::2]
            ys = np.linspace(0, self.params['height'], per_side*2+1)[1::2]
            locations = [(x,y) for x in xs for y in ys]

        if method == "random":
            locations = []
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
            locations = []
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
                first_locations = locations[:n_neighbourhoods*divider]
                rest_locations = locations[n_neighbourhoods*divider:]
                random.shuffle(rest_locations)
                locations = first_locations + rest_locations[:remainder]

        return locations


    def compute_distances(self):
        """
        Compute distance from all grid cells to all schools and all
        neighbourhood objects.

        Returns:
            dict: of dicts containing all Euclidean distances
        """
        
        EPS = 1.2e-6
        distances = {}
        closest_schools = {}
        furthest_schools = {}
        closest_neighbourhoods = {}
        for x in range(self.params["width"]):

            for y in range(self.params["height"]):
                
                max_dist_school = 0
                min_dist_school = 10e10
                loc_distances = {}

                # Loop over all neighbourhoods, calculate distance and save
                # closest
                point = Point(x,y)
                for neighbourhood in self.get_agents("neighbourhoods"):
                    shape = neighbourhood.shape.buffer(EPS)
                    if shape.contains(point):
                        closest_neighbourhoods[str((x,y))] = str(neighbourhood.pos)
                        break

                # Loop over all schools, calculate distance and save furthest
                # and closest
                for school in self.get_agents("schools"):

                    distance = self.calculate_distance(school.pos,(x, y))
                    loc_distances[str(school.pos)] = distance

                    if distance < min_dist_school:
                        min_dist_school = distance
                        closest_school = str(school.pos)

                    if distance > max_dist_school:
                        max_dist_school = distance
                        furthest_school = str(school.pos)

                closest_schools[str((x,y))] = (closest_school, min_dist_school)
                furthest_schools[str((x,y))] = (furthest_school, max_dist_school)
                distances[str((x,y))] = loc_distances

        return distances, closest_schools, furthest_schools, \
            closest_neighbourhoods


    def compute_norm_distances(self, distances, closest_schools, 
        furthest_schools, closest_neighbourhoods):
        """
        Compute normalised distance from all grid cells to all schools and all
        neighbourhood objects.

        Args:
            distances (dict): distances per position to every school/neighb.
            closest_schools (dict): closest schools per position
            furthest_schools (dict): furthest schools per position
            closest_neighbourhoods (dict): closest neighbourhood per position

        Returns:
            dict: of dicts containing all normalised school distances

        Note:
            * Neighbourhood distances are not normalised!
        """
        norm_dists = {}
        for pos in distances.keys():
            normalised = {}
            for location in distances[pos].keys():
                
                if (location!='closest_school' and location!='furthest_school'):
                    
                    # Get the min and max distance to a school and normalise
                    minimum = closest_schools[pos][1]
                    maximum = furthest_schools[pos][1]
                    normalised[location] = (maximum - distances[pos][location]) / \
                        (maximum - minimum)

                else:
                    normalised[location] = distances[pos][location]

            norm_dists[pos] = normalised
        return norm_dists


    def calculate_distance(self, pos_1, pos_2):
        """
        Calculates the distance between two points, accounting for toroidal space.
        This function is borrowed from the ContinuousSpace class definition
        by MESA.

        Args:
            pos_1 (tuple): (x,y) coordinates.
            pos_2 (tuple): (x,y) coordinates.

        Returns:
            float: Euclidean distance between the two points.

        """
        x1, y1 = pos_1
        x2, y2 = pos_2

        dx = np.abs(x1 - x2)
        dy = np.abs(y1 - y2)
        if self.params["torus"]:
            dx = min(dx, self.params["width"] - dx)
            dy = min(dy, self.params["height"] - dy)
        return np.sqrt(dx * dx + dy * dy)


    def get_distances(self, pos):
        """
        Returns distances from a given position to all school and neighbourhood
        objects.

        Args:
            pos (tuple): (x,y) coordinates.

        Returns:
            list: containing all the Euclidean distances.
        """
        return self.distances[str(pos)]


    def get_norm_distances(self, pos):
        """
        Returns nearness from a given position to all school and neighbourhood
        objects.

        Args:
            pos (tuple): (x,y) coordinates.

        Returns:
            list: containing all the normalized Euclidean distances.
        """
        return self.norm_distances[str(pos)]


    def get_agents(self, type):
        """
        Returns list of agents of given type.

        Args:
            type (str): either 'School', 'Neighbourhood', 'Household' or
            'Student'.

        Returns:
            list: containing all the objects of the specified type.
        """
        return self.agents[type]


    def export_data(self, export=False):
        """
        Export data for visualization.
        """
        if export:
            self.measurements.export_data()


    def get_shock(self, N=1):
        """
        Returns a small random value around zero used as perturbation in value
        determinations to make deterministic processes random.

        Checks if shocks are available. Returns the front value, creates a
        new array of shocks if not available.
        """

        # If empty
        sigma, size = 0.01, 1000
        if len(self.shocks) <= N:
            self.shocks = np.random.normal(0, scale=sigma, size=size)

        # Pop front value
        val = self.shocks[0:N]
        self.shocks = self.shocks[N:-1]
        return val


    def get_uniform_shock(self, N=1):
        """
        Returns a small random value around zero used as perturbation in value
        determinations to make deterministic processes random.

        Checks if shocks are available. Returns the front value, creates a
        new array of shocks if not available.
        """

        # If empty
        size = 1000
        if len(self.uniform_shocks) <= N:
            self.uniform_shocks = np.random.random(size=size)

        # Pop front value
        val = self.uniform_shocks[0:N]
        self.uniform_shocks = self.uniform_shocks[N:-1]
        return val


    def close_school(self, choice="most_segregated"):
        """
        Close a school randomly, or the most segregated.

        Args:
            choice (str): either 'most_segregated' (default) or 'random'.

        """

        if choice == "most_segregated":
            segregation = self.measurements.calculate_segregation(
                per_location=True)
            max_segregated = np.argmax(segregation)
            school = self.get_agents("schools")[max_segregated]
        elif choice == "random":
            school_index = np.random.randint(len(self.get_agents("schools")))
            school = self.get_agents("schools")[school_index]
        else:
            print("School closure method is not supported")
            sys.exit(1)

        self.get_agents("schools").remove(school)
        self.scheduler.remove(school)
        self.grid.remove_agent(school)


    def increment_agent_count(self):
        """
        Increment agent count by one.
        """
        self.agents["amount"] += 1
