# import os
# os.system('python3 compass/visualisation.py')

import sys
from bokeh.plotting import curdoc

sys.path.insert(0, "compass")
from visualisation import BokehServer

server = BokehServer(curdoc())
server.run_visualisation()