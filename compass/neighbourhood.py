"""
The Neighbourhood class.
"""
import numpy as np
from typing import List, ClassVar
from .agents_base import BaseAgent


class Neighbourhood(BaseAgent):
    """
    The Neighbourhood class.

    Args:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.

    Attributes:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.
        total (int): the number of Households in the neighbourhood.
        households (list): all the households living in the neighbourhood.
        composition (array): the sum of the attribute arrays of all Households
            belonging to this neighbourhood.
    """

    _total_neighbourhoods: ClassVar[int] = 0
    __slots__ = ["idx"]

    def __init__(
            self,
            unique_id: int,
            pos: tuple[float, float],
            shape,  # FIXME: type?
            model: object,
            params: dict
            ):

        super().__init__(unique_id, pos, model, params)

        self.idx: int = Neighbourhood._total_neighbourhoods
        Neighbourhood._total_neighbourhoods += 1

        self.total: int  = 0
        self.shape = shape
        self.households: List['Household'] = []
        self.composition: np.ndarray = self.new_composition_array()

    def __repr__(self) -> str:
        """
        Returns:
            str: representing the unique identifier of the agent.
        """
        return f"<Neighbourhood object with unique_id: {self.unique_id}>"

    def add_household(self, household: 'Household') -> None:
        """
        Adds household (not student) object to the list of the Neighbourhood
        object.

        Args:
            household (Household): Household object.
        """
        self.total += 1
        self.composition += household.attributes
        self.households.append(household)

    def remove_household(self, household: 'Household') -> None:
        """
        Removes household (not student) object from the list of the
        Neighbourhood object.

        Args:
            household (Household): Household object.
        """
        self.total -= 1
        self.composition -= household.attributes  # TODO: zero self.composition?
        self.households.remove(household)
