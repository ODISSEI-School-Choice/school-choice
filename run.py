import sys
from bokeh.plotting import curdoc

from compass.visualisation import BokehServer

server = BokehServer(curdoc())
server.run_visualisation()
