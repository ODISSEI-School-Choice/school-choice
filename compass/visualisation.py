from re import I
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os.path import dirname, join
from tornado import gen
import numpy as np

# Bokeh imports
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.transform import linear_cmap
from bokeh.application import Application
from bokeh.tile_providers import get_provider
from bokeh.layouts import row, column, gridplot
from bokeh.document import without_document_lock
from bokeh.application.handlers.function import FunctionHandler
from bokeh.models import (ColumnDataSource, Button, Select, Slider, CDSView,
                          BooleanFilter, HoverTool, Dropdown, Div, Band,
                          WheelZoomTool)

from .parameters import FLAGS
from .model import CompassModel


class BokehServer():
    """
    Class to run the Bokeh server for visualisation.

    Args:
        doc: a Bokeh Document instance

    Attributes:
        grid_plot: initiates the plot for the grid
        line_plot: initiates one line plot
        update: updates the plots with every time step
        visualise_model: starts the whole visualisation
        grid_values: obtains all the grid values from Mesa.grid
    """

    def __init__(self, doc):

        # This is important! Save curdoc() to make sure all threads
        # see the same document.
        self.doc = doc

        # Initialise a CompassModel model with the default setting.
        self.model = CompassModel(**vars(FLAGS))
        self.params = self.model.params
        self.res_ended = False
        self.school_ended = False
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Initialise layout and populate the plots with the initial
        # configuration
        self.reset_data()
        self.layout()
        print('Visualisation running...')

    def agent_filter(self, source, agent_type):
        """
        This function creates a boolean agent filter view for the Bokeh
        visualisation instance.

        Args:
            source (CDS): ColumnDataSource from Bokeh.
            agent_type (str): the agent type to be filtered.

        Returns:
            view: the view that can be used in plots.
        """

        by_type = [True if agent == agent_type else False \
                    for agent in source.data['agent_type']]
        view = CDSView(source=source, filters=[BooleanFilter(by_type)])
        return view

    def customise_grid(self):
        """
        Grid customisation; agent colors, neighbourhood boundaries and schools.

        Todo:
            size should come from a parameter, as they do not scale properly
            with grid size.
        """

        # Agent colours, agent tooltips and grid initialisation
        mapper = linear_cmap(field_name='group1',
                             palette=['blue', 'red'],
                             low=0,
                             high=1)
        TOOLTIPS = [("Residential utility", "@res_utility"),
                    ('Local composition', '@local_comp'),
                    ('Neighbourhood composition', '@n_comp'),
                    ("School utility", "@school_utility"),
                    ('Distance', '@dist_school'),
                    ('School composition', '@s_comp'),
                    ('School composition utility', '@school_comp_utility')]
        hover = HoverTool(names=["households", "schools"], tooltips=TOOLTIPS)
        self.grid = figure(x_range=(self.model.grid.x_min - 1,
                                    self.model.grid.x_max),
                           y_range=(self.model.grid.y_min - 1,
                                    self.model.grid.y_max),
                           tools=[hover, 'tap', 'pan',
                                  WheelZoomTool()],
                           tooltips=TOOLTIPS)

        # Set WheelZoomTool active by default if not lattice
        if self.params['case'].lower() != 'lattice':
            self.grid.toolbar.active_scroll = self.grid.select_one(
                WheelZoomTool)

        # Add a background map using OpenStreetMap (Google Maps is too
        # computationally expensive and cannot zoom properly)
        self.grid.add_tile(get_provider('OSM'))

        self.grid.axis.visible = False
        self.grid.grid.visible = False
        # Function to highlight all households that are currently enrolled in
        # the same school.
        self.source.selected.on_change("indices", self.select_households)

        # Plot households
        self.grid.circle(x="x",
                         y="y",
                         size=5,
                         view=self.household_view,
                         source=self.source,
                         fill_color=mapper,
                         line_color='black',
                         alpha=0.8,
                         nonselection_fill_alpha=0.2,
                         selection_fill_alpha=1,
                         name='households')

        # Plot schools
        self.grid.circle(x="x",
                         y="y",
                         size=7,
                         source=self.source,
                         view=self.school_view,
                         fill_color='yellow',
                         line_color='black',
                         name='schools')

        # Plot neighbourhoods
        self.grid.patches('x',
                          'y',
                          source=self.source,
                          view=self.neighbourhood_view,
                          fill_color=None,
                          line_color="black",
                          line_width=2,
                          hover_alpha=0,
                          hover_line_color=None,
                          name='neighbourhoods',
                          selection_fill_alpha=0)

    def init_grid_plot(self):
        """
        Initiates the grid plot for the visualisation.
        """

        # Create filters to plot households and schools sequentially
        self.household_view = self.agent_filter(self.source, 'household')
        self.school_view = self.agent_filter(self.source, 'school')
        self.neighbourhood_view = self.agent_filter(self.source,
                                                    'neighbourhood')
        self.customise_grid()

    def select_households(self, attr, old, new):
        """
        This function selects all households that are in the same school.

        Args:
            attr: -
            old: -
            new (list): indices that are clicked (can be multiple)
        """
        index = new[0]
        school_id = self.data.iloc[index].school_id
        # List all agents that have the same school id
        same_school = self.data[self.data.school_id == school_id].index
        self.source.selected.indices = list(same_school)

    def dropdown_select(self, event):
        """
        This function creates a dropdown of all schools.

        Args:
            event (str): the selected school
        """

        school_id = int(event.item)
        same_school = self.data[self.data.school_id == school_id].index
        self.source.selected.indices = list(same_school)

    def school_dropdown_func(self, width=100):
        """
        Event listener for the dropdown of all schools.
        """
        schools = self.data[self.data.agent_type == 'school']
        school_locs = [str(num) for num in range(len(schools))]
        self.school_dropdown = Dropdown(label='Select School',
                                        menu=school_locs,
                                        width=width)
        self.school_dropdown.on_click(self.dropdown_select)

    def init_line_plot(self, width=200, height=200, mode='fixed'):
        """
        Initiates the line plot for the server.
        """

        # Create a ColumnDataSource that can be updated at every step.
        TOOLTIPS = [
            ("Residential utility", "@res_utility"),
            ("Residential segregation", "@res_seg"),
            ("School utility", "@school_utility"),
            ("School segregation", "@school_seg"),
        ]
        self.plot = figure(tooltips=TOOLTIPS,
                           y_range=(0, 1),
                           plot_width=width,
                           sizing_mode=mode,
                           title="Neighbourhood/school utility/segregation")

        plot_pars = {
            'Residential utility': {
                'y': 'res_utility',
                'color': 'green',
                'lower': 'res_q5',
                'upper': 'res_q95'
            },
            'Residential segregation': {
                'y': 'res_seg',
                'color': 'blue',
                'lower': None,
                'upper': None
            },
            'School utility': {
                'y': 'school_utility',
                'color': 'orange',
                'lower': 'school_q5',
                'upper': 'school_q95'
            },
            'School segregation': {
                'y': 'school_seg',
                'color': 'purple',
                'lower': None,
                'upper': None
            }
        }

        # Plot lines, markers and bands (for utility only)
        for label in plot_pars.keys():
            y = plot_pars[label]['y']
            color = plot_pars[label]['color']
            lower = plot_pars[label]['lower']
            upper = plot_pars[label]['upper']
            self.plot.line(x='time',
                           y=y,
                           source=self.line_source,
                           line_width=2,
                           color=color,
                           legend_label=label)
            self.plot.circle(x='time',
                             y=y,
                             source=self.line_source,
                             size=5,
                             color=color,
                             legend_label=label)

            if lower is not None:
                band = Band(base='time',
                            lower=lower,
                            upper=upper,
                            source=self.line_source,
                            fill_alpha=0.2,
                            fill_color=color)
            self.plot.add_layout(band)

        self.plot.legend.location = 'top_left'

    def composition_data(self, agent_type='school'):
        """
        Calculates the composition data for schools or neighbourhoods.

        Args:
            agent_type (str): either 'school' or 'neighbourhood'

        Returns:
            DataFrame of fraction blues and reds per agent.
        """

        # Loop over all schools to get the locations and compositions
        cols = ['group0', 'group1', 'dist_school']
        data = self.data[self.data.agent_type == agent_type][cols]

        if agent_type == 'household':
            blues = data[data['group0'] == 1]['dist_school'].values
            reds = data[data['group1'] == 1]['dist_school'].values
            return {'blues': blues, 'reds': reds}
        else:
            totals = data.sum(axis=1).replace(0, 1).values
            fractions = data.div(totals, axis=0)
            blues = fractions['group0'].values
            reds = fractions['group1'].values
            return {'blues': blues, 'reds': reds}

    def init_school_composition_plot(self,
                                     width=200,
                                     height=200,
                                     mode='fixed'):
        """
        Initiates the school composition plot.
        """

        self.school_composition_quads = {}
        self.school_composition_plot = figure(title="School composition",
                                              x_range=(0, 1),
                                              plot_width=width,
                                              sizing_mode=mode)

        fractions = self.composition_data(agent_type='school')
        for group in fractions.keys():

            hist, edges = np.histogram(fractions[group], density=True, bins=20)
            self.school_composition_quads[
                group] = self.school_composition_plot.quad(
                    top=hist,
                    bottom=0,
                    left=edges[:-1],
                    right=edges[1:],
                    fill_color=group[:-1],
                    line_color="white",
                    alpha=0.7,
                    legend_label=group)

    def init_neighbourhood_composition_plot(self,
                                            width=200,
                                            height=200,
                                            mode='fixed'):
        """
        Initiates the neighbourhood composition plot.
        """
        self.neighbourhood_composition_quads = {}
        self.neighbourhood_composition_plot = figure(
            title="Neighbourhood composition",
            x_range=(0, 1),
            plot_width=width,
            sizing_mode=mode)

        fractions = self.composition_data(agent_type='neighbourhood')
        for group in fractions.keys():

            hist, edges = np.histogram(fractions[group], density=True, bins=20)
            self.neighbourhood_composition_quads[group] = \
                self.neighbourhood_composition_plot.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color=group[:-1], line_color="white", alpha=0.7,
                legend_label=group)

    def init_distribution_plot(self, width=200, height=200, mode='fixed'):
        """
        Initiates the distribution plots for residential and school utility.
        """

        self.distribution_plot = figure(title="Utility distributions",
                                        x_range=(0, 1),
                                        plot_width=width,
                                        sizing_mode=mode)

        hist_data = self.data[self.data.agent_type == 'household'][[
            'res_utility', 'school_utility'
        ]]

        # Residential utility
        hist, edges = np.histogram(hist_data.res_utility,
                                   density=True,
                                   bins=50)
        self.res_quads = self.distribution_plot.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="green",
            line_color="white",
            alpha=0.7,
            legend_label='Residential utility')

        # School utility
        if not self.residential:
            hist, edges = np.histogram(hist_data.school_utility,
                                       density=True,
                                       bins=50)
        # Do not show school utilities yet as they are all zero!
        hist = np.zeros(len(hist))
        self.school_quads = self.distribution_plot.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="orange",
            line_color="white",
            alpha=0.7,
            legend_label='School utility')

    def init_distance_plot(self, width=200, height=200, mode='fixed'):
        """
        Initiates the distance plot for the school choice process.
        """
        """
        Initiates the neighbourhood composition plot.
        """
        self.distance_quads = {}
        self.distance_plot = figure(title="Distance utility",
                                    x_range=(0, 1),
                                    plot_width=width,
                                    sizing_mode=mode)

        fractions = self.composition_data(agent_type='household')
        for group in fractions.keys():

            hist, edges = np.histogram(fractions[group], density=True, bins=20)
            self.distance_quads[group] = \
                self.distance_plot.quad(
                top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color=group[:-1], line_color="white", alpha=0.7,
                legend_label=group)

    def update_filters(self):
        """
        Updates the view filters for households, schools and neighbourhoods as
        they can change when reset is clicked (i.e., new model instance).
        """

        # Update household filter
        household_filter = [True if agent == 'household' else False for agent \
            in self.source.data['agent_type']]
        self.household_view.filters[0] = BooleanFilter(household_filter)

        # Update neighbourhood filter
        neighbourhood_filter = [True if agent == 'neighbourhood' else False for\
            agent in self.source.data['agent_type']]
        self.neighbourhood_view.filters[0] = BooleanFilter(
            neighbourhood_filter)

        # Update school filter
        school_filter = [True if agent == 'school' else False for agent in \
            self.source.data['agent_type']]
        self.school_view.filters[0] = BooleanFilter(school_filter)

    def reset_data(self):
        """
        Resets the data, could be the initial reset (new sources need to be
        created) or a subsequent one (only update the data).
        """

        # Callback object to run, step and reset the model properly.
        self.residential = True
        self.callback_obj = None
        self.data = self.model.measurements.get_bokeh_vis_data()

        # Check if it's the initial reset (create new sources) or a reset button
        # click (update .data only)
        try:
            self.source.data = self.data
            self.line_source.data = self.data[self.data.agent_type == 'system']
            self.update_filters()
            self.update_data()

        # Reset is clicked --> update .data only
        except AttributeError:
            self.source = ColumnDataSource(self.data)
            self.line_source = ColumnDataSource(
                self.data[self.data.agent_type == 'system'])

    def update_data(self):
        """
        Updates all data sources.
        """

        # Update all plots in the figure
        self.data = self.model.measurements.get_bokeh_vis_data()
        self.source.stream(self.data, len(self.data))
        self.line_source.stream(self.data[self.data.agent_type == 'system'])
        self.school_dropdown_func()

        # Update the utility histograms
        self.update_histograms()

        # Update the composition histograms
        for quads in [
                self.neighbourhood_composition_quads,
                self.school_composition_quads, self.distance_quads
        ]:

            # Grab the new data
            if quads == self.neighbourhood_composition_quads:
                hist_data = self.composition_data(agent_type='neighbourhood')
            elif quads == self.school_composition_quads:
                hist_data = self.composition_data(agent_type='school')
            else:
                hist_data = self.composition_data(agent_type='household')

            # Update the bars and edges
            for group in hist_data.keys():

                hist, edges = np.histogram(hist_data[group],
                                           density=True,
                                           bins=20)

                # Update histogram
                quads[group].data_source.data['top'] = hist
                quads[group].data_source.data['left'] = edges[:-1]
                quads[group].data_source.data['right'] = edges[1:]

    def update_histograms(self):

        hist_data = self.data[self.data.agent_type == 'household'][[
            'res_utility', 'school_utility'
        ]]

        hist, edges = np.histogram(hist_data.res_utility,
                                   density=True,
                                   bins=50)
        self.res_quads.data_source.data['top'] = hist
        self.res_quads.data_source.data['left'] = edges[:-1]
        self.res_quads.data_source.data['right'] = edges[1:]

        # Only start to show school utilities when the residential process is
        # finished
        if not self.residential:
            hist, edges = np.histogram(hist_data.school_utility,
                                       density=True,
                                       bins=50)
        else:
            hist = np.zeros(len(hist))

        self.school_quads.data_source.data['top'] = hist
        self.school_quads.data_source.data['left'] = edges[:-1]
        self.school_quads.data_source.data['right'] = edges[1:]

    def blocking_task(self):
        time.sleep(0.1)

    @without_document_lock
    @gen.coroutine
    def unlocked_task(self):
        """
        Needed to make sure that if the reset button is clicked it can go
        inbetween events, otherwise it can be quite slow.
        """
        yield self.executor.submit(self.blocking_task)
        self.doc.add_next_tick_callback(partial(self.step_button))

    def run_button(self):
        """
        Handles the run button clicks, coloring and starts the simulation.
        """
        if self.run.label == 'Run':
            self.run.label = 'Stop'
            self.run.button_type = 'danger'
            self.callback_obj = self.doc.add_periodic_callback(
                self.unlocked_task, 500)

        else:
            self.run.label = 'Run'
            self.run.button_type = 'success'
            self.doc.remove_periodic_callback(self.callback_obj)

    def step_button(self):
        """
        Checks which process need to be stepped and execute the step. The
        simulate function of the Model instance cannot be used as we need to
        visualise every step.
        """

        # Plus 1 because the first time step is already plotted
        max_res_steps = self.params['max_res_steps']
        max_school_steps = self.params['max_school_steps']
        school_time = self.model.scheduler.get_time('school')
        residential_time = self.model.scheduler.get_time('residential') + 1

        # If residential is not converged yet or below max steps, do a step
        if (residential_time <= max_res_steps and not self.res_ended):
            self.res_ended = self.model.convergence_check()
            if not self.res_ended:
                self.model.step(residential=True)
            else:
                print('Residential process ended.')

        # Initial school step needs to be executed
        elif school_time == 0:
            self.residential = False
            self.model.step(residential=False, initial_schools=True)

        # Normal school steps
        elif (school_time < max_school_steps and not self.school_ended):
            self.school_ended = self.model.convergence_check()
            if not self.school_ended:
                self.model.step(residential=False, initial_schools=False)
            else:
                print('School process ended.')

        # Both processes are done/converged
        else:
            self.run_button()
            return

        self.update_data()

    def reset_button(self):
        """
        Resets the model and takes the (possible) new parameter values into
        account.
        """

        # Update the parameter values and start a new model
        self.time_series = []
        self.res_ended = False
        self.school_ended = False
        self.model = CompassModel(self.update_pars())

        # Stop the model when it is still running while reset is clicked.
        if self.run.label == 'Stop' and self.callback_obj is not None:
            self.doc.remove_periodic_callback(self.callback_obj)
            self.run.label = 'Run'
            self.run.button_type = 'success'

        self.reset_data()
        self.doc.clear()
        self.layout()

    def layout(self):
        """
        Sets up the whole layout; widgets and all plots.
        """

        # Initialise all plots and widgets
        widgets = self.widgets(width=200)

        plot_width = 500
        sizing_mode = 'stretch_height'
        self.init_grid_plot()
        self.init_line_plot(width=plot_width, mode=sizing_mode)
        self.init_distribution_plot(width=plot_width, mode=sizing_mode)
        self.init_school_composition_plot(width=plot_width, mode=sizing_mode)
        self.init_neighbourhood_composition_plot(width=plot_width,
                                                 mode=sizing_mode)
        self.init_distance_plot(width=plot_width, mode=sizing_mode)

        # Row with widgets
        if self.params['case'].lower() == 'lattice':
            width = 420
            split = int(len(widgets) / 2.) + 1
            widget_row = row(
                [column(widgets[:split]),
                 column(widgets[split:])],
                width=width)
        else:
            width = 210
            widget_row = column(widgets, width=width)

        desc = Div(text=open(join(dirname(__file__),
                                  "description.html")).read(),
                   margin=0)
        # Column with all the controls and description
        first_col = column(widget_row, width=width, sizing_mode='fixed')

        # Column with the grid/map
        second_col = column([
            desc,
            row(self.buttons(), sizing_mode='stretch_width'),
            row(self.grid, sizing_mode='stretch_width')
        ],
                            sizing_mode='stretch_width')

        # Column with the plots
        third_col = column([
            self.plot, self.distribution_plot, self.distance_plot,
            self.school_composition_plot, self.neighbourhood_composition_plot
        ])

        # second_third = row([second_col, third_col], sizing_mode='scale_width')

        vis_layout = gridplot([[first_col, second_col, third_col]],
                              toolbar_location=None)

        self.doc.add_root(vis_layout)
        self.doc.title = "COMPASS"

    def buttons(self, width=100):
        self.run = Button(label="Run", button_type='success', height=32)
        self.run.on_click(self.run_button)
        self.step = Button(label="Step", button_type='primary', height=32)
        self.step.on_click(self.step_button)
        self.reset = Button(label="Reset", button_type='warning', height=32)
        self.reset.on_click(self.reset_button)
        buttons = [self.run, self.step, self.reset]
        return buttons

    def widgets(self, width=100):
        """
        Hardcodes all widgets.
        """

        header_size = '<h3>'

        # Simulation
        self.sleep = Slider(start=0,
                            end=1,
                            value=0.1,
                            step=0.1,
                            title='Time between steps',
                            width=width)
        self.max_move_fraction = Slider(start=0,
                                        end=1,
                                        value=self.params['max_move_fraction'],
                                        step=.05,
                                        title="Fraction moving",
                                        width=width)
        self.max_res_steps = Slider(start=0,
                                    end=1000,
                                    value=self.params['max_res_steps'],
                                    step=10,
                                    title="Residential steps",
                                    width=width)
        self.max_school_steps = Slider(start=0,
                                       end=1000,
                                       value=self.params['max_school_steps'],
                                       step=10,
                                       title="School steps",
                                       width=width)
        self.threshold = Select(title="Convergence threshold",
                                options=['0.001', '0.005', '0.01', '0.02'],
                                value=str(self.params['conv_threshold']),
                                width=width)
        self.homophily_std = Select(title="Std. of optimal fraction",
                                    options=[str(x / 100.) for x in range(11)],
                                    value=str(self.params['homophily_std']),
                                    width=width)
        self.temperature = Slider(start=0,
                                  end=500,
                                  step=1,
                                  title="Temperature",
                                  value=self.params['temperature'],
                                  width=width)
        self.window_size = Slider(start=10,
                                  end=50,
                                  value=self.params['window_size'],
                                  step=10,
                                  title="Convergence window size",
                                  width=width)
        self.case = Select(title='Case',
                           options=[
                               'Lattice',
                               'Amsterdam',
                           ],
                           value=str(self.params['case']),
                           width=width)

        if self.params['random_residential'] == 0:
            random_residential = 'False'
        else:
            random_residential = 'True'
        self.random_residential = Select(title="Random residential",
                                         options=['True', 'False'],
                                         value=random_residential,
                                         width=width)

        text = header_size + 'Simulation' + header_size
        simulation_div = Div(text=text, width=width)
        simulation = [
            simulation_div, self.max_move_fraction, self.max_res_steps,
            self.max_school_steps, self.threshold, self.homophily_std,
            self.window_size, self.temperature, self.case,
            self.random_residential
        ]

        if self.params['case'].lower() != 'lattice':
            text = header_size + 'Parameters' + header_size
            parameter_div = Div(text=text, width=width)
            simulation = [
                parameter_div, self.max_move_fraction, self.max_school_steps,
                self.threshold, self.homophily_std, self.window_size,
                self.temperature, self.case, self.random_residential
            ]

        # Grid
        self.size = Select(title="Size",
                           options=[str(x * 10) for x in range(1, 16)],
                           value=str(self.params['width']),
                           width=width)
        self.n_neighbourhoods = Select(
            title="Number of neighbourhoods",
            options=[str(x**2) for x in range(1, 16)],
            value=str(self.params['n_neighbourhoods']),
            width=width)
        self.n_schools = Select(title="Number of schools",
                                options=[str(x**2) for x in range(1, 11)],
                                value=str(self.params['n_schools']),
                                width=width)
        self.schools_placement = Select(
            title="School placement",
            options=['evenly_spaced', 'random', 'random_per_neighbourhood'],
            value=str(self.params['schools_placement']),
            width=width)
        self.household_density = Slider(start=0,
                                        end=1,
                                        value=self.params['household_density'],
                                        step=.05,
                                        title="Density",
                                        width=width)
        self.group_dist = Slider(start=0,
                                 end=1,
                                 value=self.params['group_dist'][0][0],
                                 step=.05,
                                 title="Share blue",
                                 width=width)
        self.torus = Select(title="Torus",
                            options=['True', 'False'],
                            value=str(self.params['torus']),
                            width=width)

        if self.params['scheduling'] == 0:
            scheduling = 'replacement'
        else:
            scheduling = 'without_replacement'
        self.scheduling = Select(
            title="Scheduling method",
            options=['replacement', 'without_replacement'],
            value=scheduling,
            width=width)

        text = header_size + 'Environment' + header_size
        environment_div = Div(text=text, width=width)
        grid = [
            environment_div, self.size, self.n_neighbourhoods, self.n_schools,
            self.household_density, self.group_dist, self.scheduling,
            self.torus, self.schools_placement
        ]

        if self.params['case'].lower() != 'lattice':
            grid = [self.scheduling]

        # School
        self.school_capacity = Slider(start=0,
                                      end=3,
                                      value=self.params['school_capacity'],
                                      step=.1,
                                      title="School capacity",
                                      width=width)
        self.school_dropdown_func(width=width)
        text = header_size + 'Schools' + header_size
        school_div = Div(text=text, width=width)
        school = [school_div, self.school_capacity, self.school_dropdown]

        if self.params['case'].lower() != 'lattice':
            school = [self.school_capacity, self.school_dropdown]

        # Household
        self.alpha = Slider(start=0,
                            end=1,
                            value=self.params['alpha'],
                            step=.05,
                            title="Alpha",
                            width=width)
        self.utility_at_max = Slider(start=0,
                                     end=1,
                                     value=self.params['utility_at_max'][0][0],
                                     step=.05,
                                     title="Utility at homogeneity",
                                     width=width)
        self.optimal_fraction = Slider(
            start=0,
            end=1,
            value=self.params['optimal_fraction'][0][0],
            step=.05,
            title="Optimal fraction",
            width=width)
        self.radius = Select(title="Radius",
                             options=[str(x) for x in range(1, 11)],
                             value=str(self.params['radius']),
                             width=width)
        self.neighbourhood_mixture = Slider(
            start=0,
            end=1,
            value=self.params['neighbourhood_mixture'],
            step=.05,
            title="Neighbourhood Mixture",
            width=width)
        self.num_considered = Slider(start=1,
                                     end=10,
                                     value=self.params['num_considered'],
                                     step=1,
                                     title="Considered spots",
                                     width=width)
        self.ranking_method = Select(title="Ranking method",
                                     options=['highest', 'proportional'],
                                     value=self.params['ranking_method'],
                                     width=width)
        self.p = Slider(start=500,
                        end=10000,
                        value=self.params['p'],
                        step=500,
                        title="Location of sigmoid",
                        width=width)
        self.q = Slider(start=1,
                        end=10,
                        value=self.params['q'],
                        step=1,
                        title="Slope of sigmoid",
                        width=width)

        text = header_size + 'Households' + header_size
        household_div = Div(text=text, width=width)
        household = [
            household_div, self.alpha, self.utility_at_max,
            self.optimal_fraction, self.radius, self.neighbourhood_mixture,
            self.num_considered, self.ranking_method, self.p, self.q
        ]

        if self.params['case'].lower() != 'lattice':
            household = [
                self.alpha, self.utility_at_max, self.optimal_fraction,
                self.ranking_method, self.p, self.q
            ]

        widgets = simulation + grid + school + household

        return widgets

    def update_pars(self):
        """
        Updates the parameters.
        """

        self.params['alpha'] = self.alpha.value
        self.params['max_move_fraction'] = self.max_move_fraction.value
        self.params['max_school_steps'] = self.max_school_steps.value
        self.params['conv_threshold'] = float(self.threshold.value)
        self.params['utility_at_max'] = [[
            self.utility_at_max.value, self.utility_at_max.value
        ]]
        self.params['optimal_fraction'] = [[
            self.optimal_fraction.value, self.optimal_fraction.value
        ]]
        self.params['school_capacity'] = self.school_capacity.value
        self.params['neighbourhood_mixture'] = self.neighbourhood_mixture.value
        self.params['temperature'] = self.temperature.value
        self.params['homophily_std'] = float(self.homophily_std.value)
        self.params['window_size'] = int(self.window_size.value)
        self.params['num_considered'] = int(self.num_considered.value)
        self.params['ranking_method'] = self.ranking_method.value
        self.params['case'] = self.case.value
        self.params['p'] = self.p.value
        self.params['q'] = self.q.value

        if self.random_residential.value == 'False':
            self.params['random_residential'] = 0
        else:
            self.params['random_residential'] = 1

        if self.scheduling.value == 'replacement':
            self.params['scheduling'] = 0
        else:
            self.params['scheduling'] = 1

        if self.params['case'].lower() != 'lattice':
            self.params['max_res_steps'] = 0
            return self.params

        self.params['width'] = int(self.size.value)
        self.params['height'] = int(self.size.value)
        self.grid.x_range.end = self.params['width']
        self.grid.y_range.end = self.params['height']
        self.params['torus'] = eval(self.torus.value)
        self.params['household_density'] = self.household_density.value
        self.params['max_res_steps'] = self.max_res_steps.value
        self.params['n_neighbourhoods'] = int(self.n_neighbourhoods.value)
        self.params['n_schools'] = int(self.n_schools.value)
        self.params['schools_placement'] = self.schools_placement.value
        self.params['group_dist'] = [[
            self.group_dist.value, 1 - self.group_dist.value
        ]]

        return self.params

    def run_visualisation(self):

        apps = {'/': Application(FunctionHandler(BokehServer))}
        server = Server(apps, port=5004)
        # To avoid bokeh's logger spamming
        log = logging.getLogger('bokeh')
        log.setLevel('ERROR')
        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
