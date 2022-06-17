"""
The Utils and Measurements class.
"""

import sys
import numpy as np
import pandas as pd
from typing import List
from .household import Household


class Utilities:
    """
    Class containing a number of measurement functions.

    Todo:
        * Maybe remove this class?
    """

    def __init__(self):
        pass


class Measurements:
    """
    Class storing the segregation measurements per step.

    Args:
        model (CompassModel): CompassModel object.

    Attributes:
        all_measurements (list): list of all measurements.
        current (dict): counts the number of happy and unhappy agents.
        model (CompassModel): CompassModel object.
        agents (list): all the agents in the model
        vis_data (dict): dictionary for visualisation purposes.

    """

    def __init__(self, model):

        self.all_measurements: List[Object] = []
        self.model: "CompassModel" = model
        self.params: dict = model.params
        self.agents: dict[str, List[object]] = model.agents
        self.vis_data: dict = dict()

    def headers(self):
        """
        This function creates numpy arrays with the names of the columns for
        the household, neighbourhood and school data.
        """
        self.household_headers: np.ndarray = np.array(
            [
                "loc_x",
                "loc_y",
                "local_comp_0",
                "local_comp_1",
                "utility",
                "category",
                "id",
                "distance",
                "unit",
            ]
        )
        self.neighbourhood_headers: np.ndarray = np.array(
            ["comp_0", "comp_1", "utility", "satisfied", "distance", "unit"]
        )
        self.school_headers: np.ndarray = np.array(
            ["comp_0", "comp_1", "utility", "satisfied", "distance", "unit"]
        )

    def measurement_arrays(self) -> None:
        """
        Initialises the measurement arrays.
        """
        dtype = "float32"
        # Maximum steps and number of attributes per array
        max_steps = self.params["max_res_steps"] + 1 + self.params["max_school_steps"]
        self.headers()
        self.n_household_attrs: int = len(self.household_headers)
        self.n_neighbourhood_attrs: int = len(self.neighbourhood_headers)
        self.n_school_attrs: int = len(self.school_headers)
        self.temp_household: np.ndarray = np.zeros(self.n_household_attrs, dtype=dtype)

        # Determine the maximum shape of the arrays
        households_shape = (
            max_steps,
            self.params["n_households"],
            self.n_household_attrs,
        )
        neighbourhoods_shape = (
            self.params["max_res_steps"] + 1,
            self.params["n_neighbourhoods"],
            self.n_neighbourhood_attrs,
        )
        schools_shape = (
            self.params["max_school_steps"],
            self.params["n_schools"],
            self.n_school_attrs,
        )

        # Initialise empty arrays (datatype is important)
        self.households: np.ndarray = np.zeros(shape=households_shape, dtype=dtype)
        self.neighbourhoods: np.ndarray = np.zeros(
            shape=neighbourhoods_shape, dtype=dtype
        )
        self.schools: np.ndarray = np.zeros(shape=schools_shape, dtype=dtype)

    def household_data(self, residential: bool, time: int) -> None:
        """
        Gets the required data from all the households in the model.

        Args:
            residential (bool): True if we are in the residential process
            time (int): time step we are at

        Note:
            Double check neighbourhood and school indices (from unit) and the
            data
        """

        # Constant data
        if time == 0:
            for household in self.agents["households"]:
                self.households[:, household.idx, 0] = household.pos[0]
                self.households[:, household.idx, 1] = household.pos[1]
                self.households[:, household.idx, 2] = household.composition[0]
                self.households[:, household.idx, 3] = household.composition[1]
                self.households[:, household.idx, 5] = household.category
                self.households[:, household.idx, 6] = household.unique_id

        # Dynamic data
        if residential:
            self.households[time, :, 4] = Household._household_res_utility[:]
            self.households[time, :, 7] = Household._household_distance[:]
            for household in self.agents["households"]:
                self.households[
                    time, household.idx, 8
                ] = household.neighbourhood.unique_id
        else:
            self.households[time, :, 4] = Household._household_school_utility[:]
            self.households[time, :, 7] = Household._household_distance[:]
            self.households[time, :, 8] = Household._household_school_id[:]

    def neighbourhood_data(self, time: int) -> None:
        """
        Gets the missing data from all the neighbourhoods in the model.

        Args:
            time (int): time step we are at
        """
        for idx, neighbourhood in enumerate(self.agents["neighbourhoods"]):
            self.neighbourhoods[time, idx, :2] = neighbourhood.composition
            self.neighbourhoods[time, idx, 5] = neighbourhood.unique_id

    def school_data(self, time: int) -> None:
        """
        Gets the missing data from all the schools in the model.

        Args:
            time (int): time step we are at

        Note:
            time is different for schools compared to neighbourhoods!
        """

        for idx, school in enumerate(self.agents["schools"]):
            self.schools[time, idx, :2] = school.composition
            self.schools[time, idx, 5] = school.unique_id

    def end_step(self, residential: bool) -> None:
        """
        Perform end of cycle data collection. At the end of a cycle, every
        measurement is performed. The current measurements are only for
        households and are stored in a Numpy array with datatype uint16.
        """
        # Fill arrays with data
        self.residential = residential
        time = self.model.scheduler.get_time()
        if time == 0:
            self.measurement_arrays()
        self.household_data(residential, time)
        if residential:
            self.neighbourhood_data(time)
        else:
            self.school_data(self.model.scheduler.get_time("school") - 1)

    def get_last(self) -> object:
        """
        Returns:
            int: measurements from last performed step.
        """
        return self.all_measurements[-1]

    def get_bokeh_vis_data(self) -> object:
        """
        Stores model data in the correct format for the Bokeh visualisation.

        Returns:
            DataFrame of all the Bokeh visualisation data.

        Note:
            * For simplicity it is now in one Pandas DataFrame, for easy
            updating/filtering/selecting on the Bokeh server.
        """
        # Collect data from all the different agent types
        household_data = self.vis_household_data()
        school_data = self.vis_school_data(household_data)
        neighbourhood_data = self.vis_neighbourhood_data(household_data)
        system_data = self.vis_system_data(
            household_data, school_data, neighbourhood_data
        )
        vis_data = pd.concat(
            [household_data, school_data, neighbourhood_data, system_data],
            ignore_index=True,
        )
        # Incorporate the time step of the simulation
        vis_data["time"] = np.repeat(self.model.scheduler.get_time(), len(vis_data))
        return vis_data

    def empty_dataframe(
        self, columns: List[str] = None, n_rows: int = 0
    ) -> pd.DataFrame:
        """
        Creates an empty Pandas Dataframe

        Args:
            columns (list): list of column names in string format.
            n_rows (int): number of rows.

        Returns:
            Empty DataFrame for all the Bokeh visualisation data.
        """
        if columns is None:
            columns = []
        return pd.DataFrame(index=range(n_rows), columns=columns)

    def vis_composition_data(self, household: Household) -> List[np.ndarray]:
        """
        Extracts the composition data from the households.

        Args:
            household: a Household object
        """
        try:
            school_comp = household.students[0].school.composition
        except AttributeError:
            school_comp = household.new_composition_array()
        return [
            household.composition,
            household.neighbourhood.composition,
            school_comp,
            household.school_utility_comp,
        ]

    def vis_household_data(self) -> dict[str, object]:
        """
        Transforms the household data to be suitable for the Bokeh visualisation.

        Returns:
            DataFrame of all the household data.
        """

        # Grab the different times
        time = self.model.scheduler.get_time()
        res_time = self.model.scheduler.get_time("residential")

        households = self.agents["households"]
        columns = [
            "agent_type",
            "x",
            "y",
            "group0",
            "group1",
            "res_id",
            "res_utility",
            "res_happy",
            "school_id",
            "dist_school",
            "school_utility",
            "school_happy",
            "res_q5",
            "res_q95",
            "school_q5",
            "school_q95",
            "res_seg",
            "school_seg",
            "local_comp",
            "n_comp",
            "s_comp",
            "school_comp_utility",
        ]
        data = self.empty_dataframe(columns=columns, n_rows=len(households))

        # Save location and local composition per group type
        data["agent_type"] = "household"
        data["x"] = self.households[time, :, 0]
        data["y"] = self.households[time, :, 1]
        data["group0"] = (self.households[time, :, 5] == 0).astype(int)
        data["group1"] = (self.households[time, :, 5] == 1).astype(int)

        # Neighbourhood ID, current residential utility
        data["res_id"] = self.households[res_time, :, 8]
        data["res_utility"] = self.households[res_time, :, 4]
        data["res_happy"] = None

        composition_data = pd.DataFrame(
            [self.vis_composition_data(household) for household in households]
        )
        data[
            ["local_comp", "n_comp", "s_comp", "school_comp_utility"]
        ] = composition_data

        # Fill school data if applicable, set them to zero otherwise
        if not self.residential:
            data["school_id"] = (
                self.households[time, :, 8] - self.params["n_neighbourhoods"]
            ).astype(int)
            data["dist_school"] = self.households[time, :, 7]
            data["school_utility"] = self.households[time, :, 4]
            data["school_happy"] = None
        else:
            data["school_utility"] = 0
            data["dist_school"] = 0

        return data

    def vis_school_data(self, household_data: pd.DataFrame):
        """
        Gets the required data from all the schools in the model.

        Args:
            household_data (DataFrame): all the household data already gathered.

        Returns:
            DataFrame of all the school data.

        Note:
            School data is still calculated per household and not per student
        """
        schools = self.agents["schools"]
        data = self.empty_dataframe(columns=household_data.columns, n_rows=len(schools))

        for index, school in enumerate(schools):
            agent_type = "school"
            x, y = school.pos
            group0, group1 = school.composition.astype(int)
            res_id = None
            res_utility = None
            res_happy = None

            # School attributes

            # Subtract the number of neighbourhoods for the visualisation of
            # school composition plot.
            school_id = int(school.unique_id - self.params["n_neighbourhoods"])
            pupils = household_data[household_data.school_id == school_id]
            dist_school = pupils.dist_school.mean()
            school_utility = pupils.school_utility.mean()
            school_happy = pupils.school_happy.mean()
            school_comp_utility = pupils.school_comp_utility.mean()

            res_q5, res_q95, school_q5, school_q95, res_seg, school_seg = [None] * 6
            local_comp, n_comp, s_comp = [None] * 3
            s_comp = school.composition.astype(int)

            # Add data to the DataFrame
            data.iloc[index] = [
                agent_type,
                x,
                y,
                group0,
                group1,
                res_id,
                res_utility,
                res_happy,
                school_id,
                dist_school,
                school_utility,
                school_happy,
                res_q5,
                res_q95,
                school_q5,
                school_q95,
                res_seg,
                school_seg,
                local_comp,
                n_comp,
                s_comp,
                school_comp_utility,
            ]

        return data

    def vis_neighbourhood_data(self, household_data: pd.DataFrame):
        """
        Gets the required data from all the neighbourhood in the model.

        Args:
            household_data (DataFrame): all the household data already gathered.

        Returns:
            DataFrame of all the neighbourhood data.
        """
        neighbourhoods = self.agents["neighbourhoods"]
        data = self.empty_dataframe(
            columns=household_data.columns, n_rows=len(neighbourhoods)
        )

        for index, neighbourhood in enumerate(neighbourhoods):
            agent_type = "neighbourhood"
            group0, group1 = neighbourhood.composition.astype(int)
            res_id = index
            households = household_data[
                household_data.res_id == neighbourhood.unique_id
            ]
            res_utility = households.res_utility.mean()
            res_happy = households.res_happy.mean()
            x, y = neighbourhood.pos

            if neighbourhood.shape.type == "Polygon":
                x, y = neighbourhood.shape.exterior.coords.xy
            elif neighbourhood.shape.type == "MultiPolygon":
                x, y = neighbourhood.shape.convex_hull.exterior.coords.xy

            x, y = list(x), list(y)

            # School attributes
            school_id, dist_school, school_utility, school_happy = [None] * 4
            res_q5, res_q95, school_q5, school_q95, res_seg, school_seg = [None] * 6
            local_comp, n_comp, s_comp, school_comp_utility = [None] * 4

            # Add data to the DataFrame
            data.iloc[index] = [
                agent_type,
                x,
                y,
                group0,
                group1,
                res_id,
                res_utility,
                res_happy,
                school_id,
                dist_school,
                school_utility,
                school_happy,
                res_q5,
                res_q95,
                school_q5,
                school_q95,
                res_seg,
                school_seg,
                local_comp,
                n_comp,
                s_comp,
                school_comp_utility,
            ]
        return data

    def vis_system_data(
        self,
        household_data: pd.DataFrame,
        school_data: pd.DataFrame,
        neighbourhood_data: pd.DataFrame,
    ):
        """
        Gets the required data from the whole system.

        Args:
            household_data (DataFrame): all the household data already gathered.
            school_data (DataFrame): all the school data already gathered.
            neighbourhood_data (DataFrame): all the neighbourhood data already
                gathered.

        Returns:
            DataFrame of all the system data.
        """
        data = self.empty_dataframe(columns=household_data.columns, n_rows=1)
        agent_type = "system"
        x, y, group0, group1, res_id, school = 6 * [None]
        res_utility = household_data.res_utility.mean()
        res_happy = household_data.res_happy.mean()
        dist_school = household_data.dist_school.mean()
        school_utility = household_data.school_utility.mean()
        school_happy = household_data.school_happy.mean()

        res_q5 = household_data.res_utility.quantile(q=0.05)
        res_q95 = household_data.res_utility.quantile(q=0.95)
        school_q5 = household_data.school_utility.quantile(q=0.05)
        school_q95 = household_data.school_utility.quantile(q=0.95)
        res_seg = self.calculate_segregation(
            type="bounded_neighbourhood", index="Theil"
        )
        if self.residential:
            school_seg = 0
        else:
            school_seg = self.calculate_segregation(type="school", index="Theil")

        local_comp, n_comp, s_comp = [None] * 3
        school_comp_utility = household_data.school_comp_utility.mean()

        # Add data to the DataFrame
        data.iloc[0] = [
            agent_type,
            x,
            y,
            group0,
            group1,
            res_id,
            res_utility,
            res_happy,
            school,
            dist_school,
            school_utility,
            school_happy,
            res_q5,
            res_q95,
            school_q5,
            school_q95,
            res_seg,
            school_seg,
            local_comp,
            n_comp,
            s_comp,
            school_comp_utility,
        ]
        return data

    def export_data(self):
        """
        Export the data using numpy save.
        """

        if self.model.export:
            end_time = self.model.scheduler.get_time()
            res_end_time = self.model.scheduler.get_time("residential")
            if res_end_time == 0:
                res_end_time = 1
            school_end_time = self.model.scheduler.get_time("school")

            filename = self.model.params["filename"]
            if self.model.params["save_last_only"]:
                start = end_time - 1
                res_start = res_end_time - 1
                school_start = school_end_time - 1
                households = self.households[[res_start, start], :, :]

            else:
                res_start = 0
                school_start = 0
                households = self.households[:end_time, :, :]

            print("Saving data...")

            np.savez(
                filename,
                households=households,
                chosen_indices=self.model.chosen_indices,
                households_headers=self.household_headers,
                neighbourhoods=self.neighbourhoods[res_start:res_end_time, :, :],
                neighbourhoods_headers=self.neighbourhood_headers,
                schools=self.schools[school_start:school_end_time, :, :],
                schools_headers=self.school_headers,
                params=self.params,
            )
            print("Data saved!")

    def calculate_segregation(
        self, type: str = "school", index: str = "Theil", per_location: bool = False
    ):
        """
        Calculate segregation index for the whole system.

        Args:
            type (str): calculates 'school' (defaul), 'bounded_neighbourhood'
                (Neighbhourhood) or 'local_neighbourhood' (Moore) segregation.
            index (str): 'Theil' Entropy based segregation index.
            per_location (bool): Default is False, but if True it returns the
                decomposed index.

        Returns:
            float: segregation index for the whole system.
            list: if per_location is set to True.
        """
        if index == "Theil":
            return self.calculate_theil(type, per_location)
        else:
            print("Segregation index not supported")
            exit(1)

    def calculate_theil(self, type: str, per_location: bool = False):
        """
        Calculate Theil's information index.

        Args:
            type (str): calculates 'school' (defaul), 'bounded_neighbourhood'
                (Neighbhourhood) or 'local_neighbourhood' (Moore) segregation.
            index (str): 'Theil' Entropy based segregation index.
            per_location (bool): Default is False, but if True it returns the
                decomposed index.

        Returns:
            float: segregation index for the whole system.
            list: if per_location is set to True.

        Note:
            Only works for first category in self.params["group_categories"].

        Todo:
            Decide which notation to use
        """

        # Check which composition to use
        if type == "bounded_neighbourhood":
            agents = self.agents["neighbourhoods"]
        elif type == "local_neighbourhood":
            agents = self.agents["households"]
        elif type == "school":
            agents = self.agents["schools"]
        else:
            print("Calculation of Theil's information index not supported.")
            sys.exit(1)

        pi_m = self.model.global_composition / self.model.global_composition.sum()

        local_compositions = np.empty((len(agents), len(pi_m)))
        nr_of_agents = np.empty(len(agents))
        for i, agent in enumerate(agents):
            nr_of_agents[i] = np.sum(agent.composition)
            # TODO: is there a way to move the check out of the for loop?
            if nr_of_agents[i] < 1:
                local_compositions[i] = agent.composition
            else:
                local_compositions[i] = agent.composition / nr_of_agents[i]

        total_agents = np.sum(nr_of_agents)

        pi_jm = local_compositions
        t_j = nr_of_agents
        T = total_agents
        r_jm = pi_jm / pi_m

        global_entropy = -np.sum(pi_m * np.log(pi_m))
        E = global_entropy
        log_r_jm = np.nan_to_num(np.log(r_jm))

        # Full sum if combined, leave as array if per location
        if per_location:
            H = np.sum((t_j / (T * E)) * (pi_jm * log_r_jm).T, axis=0)
        else:
            H = np.sum((t_j / (T * E)) * (pi_jm * log_r_jm).T, axis=None)
        theil = H
        return theil
