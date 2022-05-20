
# Simulation phases 

## phase1: residential=True
Household => Neighbourhood (for amsterdamincome this is done once, and fixed)

## phase2: residential=False
Students (are part of a household) => school


Measurement

# Ideas

 * make households, neighbourhoods, schools global arrays (see Measurements) such that get_households() etc. are trivial
 * turn 'list of objects' into a numpy/array of indices
 * combine Household.array_index and Household.unique_id
 * dont save all timephases? or write to disk soonish
 * remove params objects

note: Household.composition means something else depending on residential=True/False
note: the order households are processed is shuffled every timestep (requirement).
note: the order of processing students is then (indirectly) also shuffled, so lists of students can be sorted
note: number of timesteps for testrun is mostly below 300 (cut-off)
note: the number of agents could scale up a lot for fi. Londen / NL.


# Classes

## Inheritance

Agent (mesa)
 - BaseAgent agents_base.py
   - Household agents_household.py
   - School agents_spatial.py
   - Neighbourhood agents_spatial.py

Student () agents_household.py

Model (mesa)
  - CompassModel

## Attributes

Household(BaseAgent)
  const:
      array_index (int like): household index in lookup tables
      unique_id (int): unique identifier of the agent.
      model (CompassModel): CompassModel object.
      params (Argparser): containing all (agent) parameter values.
      groups (list): containing all group types.
      n_attributes (int): number of attributes
  phase1/phase2:
      pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
      composition (np array): the sum of the attribute arrays of all Households in the local composition of this household.
      normalized_composition (np array): same as above but normalized.
      students (list): the student(s) in the household.
      attributes (array): array of attributes of the specific agent.

School(BaseAgent)
  const:
      unique_id (int): unique identifier of the agent.
      pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
      model (CompassModel): CompassModel object.
      params (Argparser): containing all parameter values.
      capacity (float): the maximum amount of students that can be enrolled.
  phase1:
  phase2:
      (will have attributes in the future)
      total (int): number of students at the school
      students (list): all the Student objects enrolled in the school.
      composition (np array): the sum of the attribute arrays of all Households enrolled in this school.
      normalized_composition (np array): same as above but normalized.


Neighborhood(BaseAgent)
  const:
      unique_id (int): unique identifier of the agent.
      shape (?)
      pos (tuple): (x,y) coordinates of the agent in the 2D-grid.
      model (CompassModel): CompassModel object.
      params (Argparser): containing all parameter values.
  phase1/phase2:
      total (int): number of households in this neighbourhood
      households (list): all the households living in the neighbourhood.
      composition (np array): the sum of the attribute arrays of all Households belonging to this neighbourhood.
      normalized_composition (np array): same as above but normalized.


Student()
  const:
      unique_id (int): unique identifier of the agent.
      household (Household): Household object.
      groups (list): containing all group types.
  phase2:
      school_preference (list): a ranking of School objects.
      school (School): School object that the student is enrolled in.
      satisfied (bool): True if the student is satisfied with current school. (TODO: remove)
      school_history (list): all the school objects the student has attended.

