# Computational Modelling of Primary School Segregation (COMPASS)
This Python implementation tries to model school choice and resulting school segregation based on the work of Schelling (1971) and Stoica & Flache (2014).

**This is the folk of the master branch of the [original GitLab repo](https://gitlab.computationalscience.nl/edignum/school-choice---understanding-segregation), the dataset used can be found [here](https://surfdrive.surf.nl/files/index.php/s/MN7DfAWklDgtoYG).**


## Usage
### Packages
This project uses `Anaconda` for package handling. Use the following commands to create an environment with the correct packages.

* `conda env create -f conda_env.yml`

After running this command, a new conda environment called `mesa` will be installed, and you can open it by using `conda activate mesa`. After this, you will be able to use python as usual with the correct packages and dependencies.

### Update Documentation
Install pdoc3 if you haven't already done so. Browse to the compassproject folder in your terminal and run `pdoc3 --html --force --output-dir docs compass`. The documentation should be updated now.

### Overview
The repository consists of:
* **run.py:** a script that runs the model interactively with a visualisation (browser)
* **testrun.py:** a test script (work in progress)
* **agents_household.py:** the household and student classes
* **agents_spatial.py:** the neighbourhood and school classses
* **allocator.py:** allocates the students to their school of choice
* **agents_base.py:** overarching agent used for inheritance
* **model.py:** initialises the entire system and all of its components
* **parameters.py:** contains all the parameter values for the simulation
* **scheduler.py:** takes care of the activation, sequence and placement of all agents
* **visualisation.py:** browser based visualisation
* **utils.py:** containing all measurements

### Simulations
Information on how to run the code here.
