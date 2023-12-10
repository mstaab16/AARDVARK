import datetime

import dash
import numpy as np
from dash import Dash, dcc, html, Input, Output, callback
import plotly
import plotly.express as px

from sim_ARPES import simulate_ARPES_measurement

# pip install pyorbital
from pyorbital.orbital import Orbital
satellite = Orbital('TERRA')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('TERRA Satellite Live Feed'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=0.5*1000, # in milliseconds
            n_intervals=0
        )
    ])
)


@callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    arrs = simulate_ARPES_measurement(
                polar = np.random.uniform(-15,15),
                tilt = np.random.uniform(-15,15), 
                azimuthal = np.random.uniform(0,360),
                photon_energy=100.0, noise_level=0.1,
                acceptance_angle=30.0, num_angles=256,
                num_energies=256, temperature=30.0,
                k_resolution=0.011, e_resolution=0.025,
                energy_range=(-0.7, 0.1), random_bands=True
            )
    fig = px.imshow(np.flipud(arrs[0]), color_continuous_scale='Hot')
    # satellite = Orbital('TERRA')
    # data = {
    #     'time': [],
    #     'Latitude': [],
    #     'Longitude': [],
    #     'Altitude': []
    # }

    # # Collect some data
    # for i in range(180):
    #     time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
    #     lon, lat, alt = satellite.get_lonlatalt(
    #         time
    #     )
    #     data['Longitude'].append(lon)
    #     data['Latitude'].append(lat)
    #     data['Altitude'].append(alt)
    #     data['time'].append(time)

    # # Create the graph with subplots
    # fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2)
    # fig['layout']['margin'] = {
    #     'l': 30, 'r': 10, 'b': 30, 't': 10
    # }
    # fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    # fig.append_trace({
    #     'x': data['time'],
    #     'y': data['Altitude'],
    #     'name': 'Altitude',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # }, 1, 1)
    # fig.append_trace({
    #     'x': data['Longitude'],
    #     'y': data['Latitude'],
    #     'text': data['time'],
    #     'name': 'Longitude vs Latitude',
    #     'mode': 'lines+markers',
    #     'type': 'scatter'
    # }, 2, 1)

    return fig


if __name__ == '__main__':
    app.run(debug=True)