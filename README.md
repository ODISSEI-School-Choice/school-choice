# Computational Modelling of Primary School Segregation (COMPASS)
This Python implementation tries to model school choice and resulting school segregation based on the work of Schelling (1971) and Stoica & Flache (2014).

**This is the folk of the master branch of the [original GitLab repo](https://gitlab.computationalscience.nl/edignum/school-choice---understanding-segregation), the dataset used can be found [here](https://surfdrive.surf.nl/files/index.php/s/MN7DfAWklDgtoYG).**


## Usage
### Packages
This project uses `pipenv` for package handling. Use the following commands to create an environment with the correct packages.

* `pip3 install pipenv` - to ensure `pipenv` is correctly installed
* `pipenv install --ignore-pipfile` - to create an environment as described in `Pipfile.lock`

After installing the environment, open it by using `pipenv shell`. After this, you will be able to use python as usual with the correct packages and dependencies.

When you wish to add/remove a package, be sure to use `pipenv install/uninstall package_name` instead of `pip3 install/uninstall package_name`. Always update the lock file with `pipenv lock` after installing/uninstalling a package you wish to keep and include both `Pipfile` and `Pipfile.lock` in the git log.

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
