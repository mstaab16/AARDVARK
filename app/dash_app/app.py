import dash
# import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
import json
import base64
import io
from PIL import Image
import pickle

from db.database import get_db
from db.models import Experiment, Decision, Measurement, Data, Report
from sqlalchemy import func
from maestro_api import maestro_messages as mm

app = dash.Dash(__name__,#  external_stylesheets=[dbc.themes.BOOTSTRAP]
                ### THE ONLY CHANGE FROM THE ABOVE, PLEASE ADD IN THIS LINE
                requests_pathname_prefix='/dashboard/')

def array_to_rgb_image(data_array, colormap='Grays', vmin=None, vmax=None):
    # Normalize the data array
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Map the normalized data to RGBA values using the colormap
    scalar_map = ScalarMappable(norm=norm, cmap=colormap)
    rgba_values = scalar_map.to_rgba(data_array, alpha=None, bytes=True)

    # Convert the RGBA values to a uint8 RGB image
    rgb_image = np.uint8(256-(rgba_values[:, :, :3] * 255))
    # rgb_image = np.roll(rgb_image, 1, axis=2)
    return rgb_image

def np_image_to_base64(im_matrix):
    rgb_image = array_to_rgb_image(im_matrix, colormap='terrain', vmin=None, vmax=None)
    im = Image.fromarray(rgb_image, mode='RGB')
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

# app.layout = html.Div([
#     # Scatter plot
#     html.Div([
#         dcc.Graph(id='scatter-plot'),
#     ], style={'width': '33%', 'display': 'inline-block'}),

#     # Image 1
#     html.Div([
#         html.Img(id='image-1'),
#     ], style={'width': '33%', 'display': 'inline-block'}),

#     # Image 2
#     html.Div([
#         html.Img(id='image-2'),
#     ], style={'width': '33%', 'display': 'inline-block'}),
# ])

# # Sample image data
# # image_path_1 = 'path_to_your_image_1.png'
# # image_path_2 = 'path_to_your_image_2.png'

# # Callback to update scatter plot
# @app.callback(
#     Output('scatter-plot', 'figure'),
#     Input('scatter-plot', 'id')
# )
# def update_scatter_plot(_):
#     return {
#         'data': [
#             go.Scatter(
#                 # x=scatter_data['x'],
#                 # y=scatter_data['y'],
#                 # mode='markers',
#                 # marker=dict(size=10),
#             ),
#         ],
#         'layout': go.Layout(
#             title='Scatter Plot',
#             showlegend=False,
#         ),
#     }

# # Callbacks to update images
# @app.callback(
#     Output('image-1', 'src'),
#     Output('image-2', 'src'),
#     Input('scatter-plot', 'id')
# )
# def update_images(_):
#     # Sample image paths (replace with your paths)
#     # image_path_1 = 'path_to_your_updated_image_1.png'
#     # image_path_2 = 'path_to_your_updated_image_2.png'

#     # Encode images to base64 for display
#     # encoded_image_1 = base64.b64encode(open(image_path_1, 'rb').read()).decode('ascii')
#     # encoded_image_2 = base64.b64encode(open(image_path_2, 'rb').read()).decode('ascii')
#     # arr1 = np.random.uniform(0,1e6,(128,128)).astype(np.int32)
#     x = np.linspace(-1,1,128)
#     y = np.linspace(-1,1,128)
#     x, y = np.meshgrid(x,y)
#     arr1 = np.exp(-(x**2 + y**2))
#     # arr1 /= arr1.max()
#     # arr1 *= 1e9
#     # arr1 = arr1.astype(np.int32)

#     # arr1 = np.max(arr1)
#     encoded_image_1 = np_image_to_base64(arr1)
#     encoded_image_2 = np_image_to_base64(arr1)

#     return encoded_image_1, encoded_image_2


# scatter_fig = go.Figure(data=go.Scatter3d())
# scatter_fig.update_traces(
#     hoverinfo="none",
#     hovertemplate=None,
# )

# app.layout = html.Div([
#     html.H4('Interactive scatter plot'),
#     dcc.Input(id='experiment-id', type='number', value=1, required=True),
#     html.Div([
#         dcc.Graph(id="scatter-plot", figure=scatter_fig),
#     ]),
#     dcc.Graph(id="image"),
#     # html.P("Filter by petal width:"),
#     dcc.Interval(
#             id='interval-component',
#             interval=1*1000, # in milliseconds
#             n_intervals=0
#         ),
# ])

# @app.callback(
#     Output("scatter-plot", "figure"), 
#     Input('interval-component', 'n_intervals'),
#     Input('experiment-id', 'value'),
# )
# def update_bar_chart(n, experiment_id):
#     try:
#         db_gen = get_db()
#         db = next(db_gen)
#         query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
#         df = pd.read_sql(query.statement, query.session.bind)
#         positions=pd.json_normalize(df['positions'])
#         fig = go.Figure(data=go.scatter3d(
#             positions, x="motors::X", y="motors::Y", 
#             width=500, height=500,
#             # color="species", size='petal_length', 
#             hover_data=["thumbnail"])
#         )
#         fig.update_xaxes(range=[0, 1])
#         fig.update_yaxes(range=[0, 1])
#         return fig
#     except:
#         fig = px.scatter(width=500, height=500)
#         fig.update_xaxes(range=[-10, 10])
#         fig.update_yaxes(range=[-10, 10])
#         return fig
    
# @app.callback(
#     Output("graph-tooltip-5", "show"),
#     Output("graph-tooltip-5", "bbox"),
#     Output("graph-tooltip-5", "children"),
#     Input("graph-5", "hoverData"),
#     Input('experiment-id', 'value'),
# )
# def display_hover(hoverData, experiment_id):
#     if hoverData is None:
#         return False, no_update, no_update

#     # demo only shows the first point, but other points may also be available
#     img_pickled = hoverData["thubmnail"]
#     bbox = hoverData["bbox"]
#     img_matrix = pickle.loads(img_pickled)

#     im_url = np_image_to_base64(img_matrix)
#     children = [
#         html.Div([
#             html.Img(
#                 src=im_url,
#                 style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
#             ),
#         ])
#     ]

#     return True, bbox, children

# @app.callback(
#     Output("image", "figure"), 
#     Input('interval-component', 'n_intervals'),
#     Input('experiment-id', 'value'),
# )
# def update_bar_chart(n, experiment_id):
#     try:
#         db_gen = get_db()
#         db = next(db_gen)
#         data, data_info = db.query(Data.data, Data.data_info).filter(Data.experiment_id == experiment_id, Data.fieldname == "Fixed_Spectra0").order_by(Data.measurement_id.desc()).first()
#         data_info = json.loads(data_info)
#         data =  base64.decodebytes(data)
#         data = np.frombuffer(data, dtype=np.int32).reshape(*data_info['dimensions'])
#         fig = px.imshow(data)
#     except Exception as e:
#         print(e)
#         # pass
#         fig = px.imshow(np.random.uniform(0,1,(8,8)), width=500, height=500)
#         # fig.update_xaxes(range=[-10, 10])
#         # fig.update_yaxes(range=[-10, 10])
#     return fig


app.layout = html.Div([
    html.H4('Experiment id'),
    dcc.Input(id='experiment-id', type='number', value=1, required=True),
    html.H4('Number of Clusters to use'),
    dcc.Input(id='n-clusters', type='number', value=1, required=True),
    html.Div([
    html.Div(dcc.Graph(id="scatter-plot"), style={'width': '33%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id="image"), style={'width': '33%', 'display': 'inline-block'}),
    ]),
    # html.P("Filter by petal width:"),
    dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        ),
    # dash_table.DataTable(id='measurement-table', data=[]),
    # html.Div([dash_table.DataTable(
    #             id='measurement-table',
    #             data=[],
    #             # columns=[{"name": i, "id": i} for i in df.columns],
    #             # fixed_rows={'headers': True, 'data': 0},
    #             # fixed_columns={'headers': True, 'data': 1},
    #             # export_columns='visible',
    #             # export_format='xlsx')
    # )], style={'border':'2px grey solid'})
])

# @app.callback(
#     Output('measurement-table', 'data'),
#     Input('interval-component', 'value'),
#     Input('experiment-id', 'value'),
# )
# def update_table_data(n, experiment_id):
#     db = get_db().__next__()
#     query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id).order_by(Measurement.ai_cycle.desc())
#     df = pd.read_sql(query.statement, query.session.bind)
#     # positions=pd.json_normalize(df['positions'])

#     return df.to_dict('records')

# @app.callback(
#     Output('measurement-table', 'columns'),
#     Input('interval-component', 'value'),
#     Input('experiment-id', 'value'),
# )
# def update_table_cols(n, experiment_id):
#     db = get_db().__next__()
#     query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id).order_by(Measurement.ai_cycle.desc())
#     df = pd.read_sql(query.statement, query.session.bind)

#     return [{"name": i, "id": i} for i in df.columns]


@app.callback(
    Output("scatter-plot", "figure"), 
    Input('interval-component', 'n_intervals'),
    Input('experiment-id', 'value'),
    Input('n-clusters', 'value'),
)
def update_bar_chart(n, experiment_id, n_clusters):
    try:
        db_gen = get_db()
        db = next(db_gen)
        # experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
        # experiment.n_clusters = n_clusters
        # db.commit()
        query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
        df = pd.read_sql(query.statement, query.session.bind)
        positions=pd.json_normalize(df['positions'])
        fig = px.scatter(
            positions, x="motors::X", y="motors::Y", 
            width=500, height=500,
            # color="species", size='petal_length', 
            # hover_data=['petal_width'])
        )
        # fig.update_xaxes(range=[0, 1])
        # fig.update_yaxes(range=[0, 1])
        return fig
    except:
        fig = px.scatter(width=500, height=500)
        fig.update_xaxes(range=[-10, 10])
        fig.update_yaxes(range=[-10, 10])
        return fig

@app.callback(
    Output("image", "figure"), 
    Input('interval-component', 'n_intervals'),
    Input('experiment-id', 'value'),
)
def update_bar_chart(n, experiment_id):
    try:
        db_gen = get_db()
        db = next(db_gen)
        data, data_info = db.query(Data.data, Data.data_info).filter(Data.experiment_id == experiment_id, Data.fieldname == "Fixed_Spectra0").order_by(Data.measurement_id.desc()).first()
        data_info = json.loads(data_info)
        data =  base64.decodebytes(data)
        data = np.frombuffer(data, dtype=np.int32).reshape(*data_info['dimensions'])
        fig = px.imshow(data.T, origin='lower')
    except Exception as e:
        print(e)
        # pass
        fig = px.imshow(np.random.uniform(0,1,(8,8)), width=500, height=500)
        # fig.update_xaxes(range=[-10, 10])
        # fig.update_yaxes(range=[-10, 10])
    return fig



if __name__ == "__main__":
    app.run_server(debug=True)