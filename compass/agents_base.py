"""
Implementation of the BaseAgent class.
"""
import numpy as np
from mesa import Agent


class BaseAgent(Agent):
    """
    Extended base class for a MESA model agent.

    Args:
        unique_id (int): unique identifier of the agent.
        pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
        model (CompassModel): CompassModel object.
        params (Argparser): containing all parameter values.

    """

    def __init__(self, unique_id, pos, model, params):

        super().__init__(pos, model)
        self.unique_id = unique_id
        self.pos = pos
        self.model = model
        self.params = params
        self.model.increment_agent_count()

    def __repr__(self):
        """
        Returns:
            str: output string representing the unique identifier of the agent.
        """
        return f"<BaseAgent object with unique_id: {self.unique_id}>"

    def step(self, **kwargs):
        """
        Perform an agent step.

        Note:
            Agents should have this method for the model to work.
        """
        pass

    def advance(self, **kwargs):
        """
        Advance scheduled step.

        Note:
            Agents should have this method for the model to work.
        """
        pass

    def new_composition_array(self):
        """
        Returns:
            array: a new composition array with all values set to zero.
        """
        n_attributes = len(self.params["group_types"][0])
        composition = np.zeros(n_attributes)
        return composition
