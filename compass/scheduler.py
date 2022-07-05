"""
The time handler. The scheduler defined in this file is equivalent to the
SimultaneousActivation scheduler from MESA v0.8.6, with the addition of
randomization within the step and advance phase plus the possibility of
parallelization over cores. The execution of the step() and advance() parts in
the agents used, result in a system that is not entirely simultaneous.
"""

from collections import OrderedDict
import numpy as np
from .allocator import Allocator


class ThreeStagedActivation:
    """
    A scheduler to simulate the simultaneous activation of all the agents.

    Args:
        model (CompassModel): CompassModel object.

    Attributes:
        model (CompassModel): CompassModel object.
        steps (int): counts the number of steps.
        time (int): counts the number of time steps.
        moved_agents (int): amount of agents moved in a time step.
        max_movements (int): maximum amount of agents to be moved per time step.
        _agents (dict): contains all the agents to be moved.
        allocater (Allocater): an Allocater object.

    Todo:
        * Parallelization if needed.
    """

    def __init__(self, model):

        self.model = model
        self.params = model.params
        (
            self.time,
            self.split_index,
            self.moved_agents,
            self.school_steps,
            self.residential_steps,
        ) = (0, 0, 0, 0, 0)

        # A fraction of 0.40 means three splits as does 0.35 for example.
        self.n_splits = np.ceil(1 / self.params["max_move_fraction"])
        self.step_size = 1 + int(self.model.params["n_households"] // self.n_splits)
        self.max_movements = 1 + int(
            self.model.params["max_move_fraction"] * self.model.params["n_students"]
        )
        self._agents = dict()
        self.allocator = Allocator()

    def add(self, agent):
        """
        Add an Agent object to the schedule.

        Args:
            agent: an Agent to be added to the schedule.

        Note:
            The agent must have a step() method.
        """
        if agent.__class__.__name__ not in self._agents:
            self._agents[agent.__class__.__name__] = OrderedDict()
        self._agents[agent.__class__.__name__][agent.unique_id] = agent

    def remove(self, agent):
        """
        Remove all instances of a given agent from the schedule.

        Args:
            agent: an Agent object.
        """
        del self._agents[agent.__class__.__name__][agent.unique_id]

    def get_agent_count(self):
        """
        Returns:
            int: the current number of agents in the queue.
        """
        return sum([len(agents) for agents in self._agents.values()])

    def get_time(self, time_type="time"):
        """
        Returns:
            int: current time step (total, residential or school)
        """

        if time_type == "time":
            return self.time
        if time_type == "residential":
            return self.residential_steps
        if time_type == "school":
            return self.school_steps

        return None

    def agents_to_move(self, households, initial_schools):
        """
        Determine which households need to be moved this step.
        """

        # Necessary because of the reset button in the visualisation, it's too
        # early in the init()
        if self.get_time() == 0:
            # A fraction of 0.40 means three splits as does 0.35 for example.
            self.n_splits = np.ceil(1 / self.params["max_move_fraction"])
            self.step_size = 1 + int(self.model.params["n_households"] // self.n_splits)
            self.max_movements = 1 + int(
                self.model.params["max_move_fraction"] * self.model.params["n_students"]
            )

        if self.params["scheduling"] == 0 or initial_schools:
            self.model.random.shuffle(households)
            self.lower_index = 0
            self.upper_index = -1

        # Determine the new splits as everyone has been stepped
        elif self.split_index % self.n_splits == 0:
            self.model.random.shuffle(households)
            self.split_index = 0
            self.lower_index = 0
            self.upper_index = self.step_size
            self.split_index += 1

        else:
            self.lower_index = self.upper_index
            self.upper_index = (1 + self.split_index) * self.step_size
            self.split_index += 1

        return households[self.lower_index : self.upper_index]

    def step(self, residential=False, initial_schools=False):
        """
        Steps all agents and then advances them.

        Args:
            residential (bool): True if a residential step needs to be
                performed, False (default) means a school step.
            initial_schools (bool): True if an initial school step needs to be
                performed, False (default) means a school step.
        """

        all_households = self.model.get_agents("households")
        households_to_move = self.agents_to_move(all_households, initial_schools)

        # FIXME: we shouldnt store this, but is needed for a test
        self.households_to_move = households_to_move  # For testing purposes

        if residential:

            # Rankings are still calculated in the Household object instead of
            # model wide as for the schools.
            for household in households_to_move:
                household.step(residential=residential, initial_schools=initial_schools)

            # UPDATE COMPOSITIONS OF ALL AGENTS AFTER ALL THE MOVES
            self.model.calc_residential_compositions()
            for household in all_households:
                household.update_residential()
            self.model.calc_res_utilities()
            self.residential_steps += 1

        else:

            if initial_schools:

                # Initial allocation for EVERY household
                # Set initial preferences
                if self.params["case"].lower() == "lattice":
                    self.model.compute_school_distances()  # execute only in lattice case
                    p = self.params["p"]
                    q = self.params["q"]
                    self.model.distance_utilities = 1.0 / (
                        1 + (self.model.all_distances / p) ** q
                    )

                #TODO: remove commented code
                #      initialization happens in new allocator
                #for household in all_households:
                #    for student in household.students:
                #        student.set_school_preference(
                #            household.school_ranking_initial()
                #        )

                # Allocate students after all initial preferences have been set
                self.allocate_schools(all_households, initial_schools)

            else:
                # Normal school step
                self.model.calc_school_rankings(
                    households_to_move, self.model.get_agents("schools")
                )
                self.allocate_schools(households_to_move, initial_schools)

            # Calculate the new school compositions:
            #  * Compass.school_compositions
            #  TODO: should this move to in the 'residential' below?
            self.model.calc_school_compositions()

            if residential:
                # Set:
                #  * Compass.neighbourhodd_compositions
                #  * Compass.local_compositions
                #  * Household.composition
                for household in all_households:
                    household.update_residential()
            # updating the household.distance is done when moving the student

            # Set:
            #  * Household.school_utility_comp
            #  * Household.school_utility
            self.model.calc_school_utilities()
            self.school_steps += 1

        # Updating of the utilities of ALL agents was done in
        # self.model.calc_school_utilities()

        self.time += 1
        self.model.measurements.end_step(residential)

    def allocate_schools(self, households, initial_schools):
        """
        Allocates students to schools.

        Args:
            households (list): list of all the households that need to be
                allocated.
            initial_schools (bool): True if an initial school step needs to be
                performed, False (default) means a school step.
        """
        if initial_schools:
            self.allocator.initial_school(households)
        else:
            self.allocator.optimal_school(households)
