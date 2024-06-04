import dash
# import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output, State

from sklearn.metrics import adjusted_mutual_info_score

import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable
import json
import base64
import io
from PIL import Image
import time

from db.database import get_db
from db.models import Experiment, Decision, Measurement, Data, Report
# from sqlalchemy import func
from maestro_api import maestro_messages as mm

app = dash.Dash(__name__)#  external_stylesheets=[dbc.themes.BOOTSTRAP]
                ### THE ONLY CHANGE FROM THE ABOVE, PLEASE ADD IN THIS LINE
                #requests_pathname_prefix='/dashboard/')

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

app.layout = html.Div([
    html.Button('Button', id='button', n_clicks=0),
    dcc.Checklist(id='checklist', options=[{'label': 'Option 1', 'value': '1'}, {'label': 'Option 2', 'value': '2'}], value=['1']),
    html.H4('Experiment id'),
    dcc.Input(id='experiment-id', type='number', value=1, required=True), 
    # html.H4('Spectrum bin factor for displaying faster'),
    # dcc.Input(id='spectrum-bin-factor', type='number', value=4, required=True),
    html.H4('Number of Clusters to use'),
    dcc.Input(id='n-clusters', type='number', value=1, required=True),
    dcc.Dropdown(options=[], id='report-type'),
    html.Div([
    html.Div(dcc.Graph(id="scatter-plot"), style={'width': '33%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id="report"), style={'width': '33%', 'display': 'inline-block'}),
    html.Div(dcc.Graph(id="image"), style={'width': '33%', 'display': 'inline-block'}),
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
    ]),
])

@app.callback(Output('button', 'children'),
              Input('button', 'n_clicks'),
              Input('experiment-id', 'value'),
              Input('report-type', 'value'),
              Input('n-clusters', 'value'),
              prevent_initial_call=True,
              )
def compute(n_clicks, experiment_id, report_name, n):
    if n != 42:
        print("NOT THE MAGIC NUMBER (42)")
        return no_update
    print("*"*20 + "THE COMPUTE IS RUNNING" + "*"*20)
    db_gen = get_db()
    db = next(db_gen)
    query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
    df = pd.read_sql(query.statement, query.session.bind)
    positions = pd.json_normalize(df['positions'])
    x = positions['motors::X']
    y = positions['motors::Y']

    query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == report_name).order_by(Report.report_id.desc()).all()
    for i in range(len(query)):
        img = np.asarray(query[i][0], dtype=np.uint8)
        n_measured = query[i][1]
        print(f"Image {i} shape: {img.shape} and n_measured: {n_measured}. Min: {img.min()} Max: {img.max()}")
        plt.imshow(img, origin='lower', extent=[x.min()-0.6, x.max(), y.min()-0.6, y.max()])
        plt.scatter(x[:n_measured], y[:n_measured], c='k', s=1, alpha=0.5)
        plt.axis('off')
        plt.savefig(f'save/imgs/{experiment_id}_{report_name}_{n_measured}_{i}.png', bbox_inches='tight')
        plt.clf()
        
    print(len(query))

    print("*"*20 + "THE COMPUTE IS DONE" + "*"*20)
    return n_clicks

# @app.callback(Output('checklist', 'options'),
#         Output('checklist', 'value'),
#         Input('experiment-id', 'value'),
#         Input('button', 'n_clicks'),
#         )
# def on_button_click(experiment_id, n_clicks):
#     print("-"*80)
#     db_gen = get_db()
#     db = next(db_gen)
#     query = json.loads(db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == experiment_id).first()[0])
#     print(query)
#     # if query[0] is None:
#     #     query = db.query(Data.fieldname).filter(Data.experiment_id == experiment_id).distinct().all()
#     #     data_names = db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == experiment_id)
#     #     data_names = json.dumps({query[0][0]: True})
#     #     db.query(Experiment).filter(Experiment.experiment_id == experiment_id).update({Experiment.data_names_to_learn: data_names})
#     #     db.commit()
#     # print(query[0][0])
#     options = []
#     value = []
#     for k, v in query.items():
#         options.append(
#             {'label': k, 'value': k},
#         )
#         if v:
#             value.append(k)
#     return options, value

@app.callback(Output('image', 'figure'),
              Input("scatter-plot", "clickData"),
              Input('image', 'figure'),
            #   Input("scatter-plot", "hoverData"),
            )
def update_figure(clickData, figure):
    start = time.perf_counter_ns()
    # print(f"{clickData=}")
    if clickData is None:
        return no_update
    
    measurement_id = clickData['points'][0]['customdata'][0]
    
    db_gen = get_db()
    db = next(db_gen)
    query = db.query(Experiment.experiment_id).order_by(Experiment.experiment_id.desc()).first()
    print(f"Experiment ID: {query.experiment_id}")
    query = db.query(Data.data, Data.data_info).filter(Data.measurement_id == measurement_id, Data.fieldname == "Fixed_Spectra5").order_by(Data.measurement_id.desc()).first()
    if query is None:
        return no_update
    data, data_info = query
    # print("DASH SEES Data: " , data)
    data_info = json.loads(data_info)
    # data =  base64.decodebytes(data)
    # print("Dash sees: ", data)
    data = np.fromfile(data, dtype=np.int32).reshape(*data_info['dimensions'], order='F')
    data = resize(data, (128,128), anti_aliasing=True)
    
    if not figure:
        figure = px.imshow(data, origin='lower', color_continuous_scale='Blues')
    else:
        figure['data'][0]['z'] = data
    print(f"Dash took {(time.perf_counter_ns()-start)/1e6:.02f}ms to update image.")
    return figure

@app.callback(Output('report', 'figure'),
            #   Input('interval-component', 'n_intervals'),
              Input('button', 'n_clicks'),
              Input('experiment-id', 'value'),
              Input('report-type', 'value'),
            #   Input("scatter-plot", "hoverData"),
            )
def update_report(n, experiment_id, report_type):
    start = time.perf_counter_ns()
    db_gen = get_db()
    db = next(db_gen)
    # query = db.query(Report).filter(Report.experiment_id == experiment_id).all()
    if report_type is None:
        query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id).order_by(Report.report_id.desc()).first()
    else:
        query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == report_type).order_by(Report.report_id.desc()).first()
    print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to query report for image EXPERIMENT ID {experiment_id}.")
    # print(query)
    if query is None:
        print("Not updating figure")
        return no_update
    image = np.asarray(query[0])

    # print(query)
    if len(image.shape) == 2:
        print('2D image')
        figure = px.imshow(image, origin='lower', color_continuous_scale='Viridis', title=f"Number of points: {query[1]}")
    elif len(image.shape) == 3:
        print('3D image')
        figure = px.imshow(image, origin='lower', title=f"Number of points: {query[1]}")
    # print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to get image.")
    # figure = px.imshow(image, origin='lower', title=f"Number of points: {query[1]}")
    # print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to make figure.")
    # else:
    #     return no_update
    print(f"Dash took {(time.perf_counter_ns()-start)/1e6:.02f}ms to update report.")
    return figure

@app.callback(Output('report-type', 'options'),
              Output('report-type', 'value'),
            #   Input('interval-component', 'n_intervals'),
              Input('button', 'n_clicks'),
              Input('report-type', 'value'),
              Input('experiment-id', 'value'),
            #   Input("scatter-plot", "hoverData"),
            )
def update_report(n, current_report_type, experiment_id):
    start = time.perf_counter_ns()
    db_gen = get_db()
    db = next(db_gen)
    query = db.query(Report.name).filter(Report.experiment_id == experiment_id).distinct()
    if query is None:
        return no_update
    print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to query.")
    options = np.ravel(query.all())
    # figure = px.imshow(query.data['image'], origin='lower', color_continuous_scale='Turbo')
    print(f"Dash took {(time.perf_counter_ns()-start)/1e6:.02f}ms to update report-type options.")
    if current_report_type not in options and options.size > 0:
        current_report_type = options[0]
    return options.tolist(), current_report_type


@app.callback(
    Output("scatter-plot", "figure"), 
    Input('interval-component', 'n_intervals'),
    Input('experiment-id', 'value'),
    Input('n-clusters', 'value'),
)
def update_scatter_plot(n, experiment_id, n_clusters):
    try:
        db_gen = get_db()
        db = next(db_gen)
        # experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
        # experiment.n_clusters = n_clusters
        # db.commit()
        query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
        df = pd.read_sql(query.statement, query.session.bind)
        # print(df.columns)
        positions=pd.json_normalize(df['positions'])
        fig = px.scatter(
            positions, x="motors::X", y="motors::Y", 
            width=500, height=500,
            # color='Red',
            hover_data = {
                'measurement_id' : df['measurement_id'],
            },
        )
        source = Image.fromarray(np.uint8(np.random.uniform(0,255,(128,128)))).convert('RGB')
        bounds = [[-10,10], [-10,10]]
        x0 = bounds[0][0]
        y0 = bounds[1][0]
        xsize = bounds[0][1] - bounds[0][0]
        ysize = bounds[1][1] - bounds[1][0]

        # fig.update_xaxes(range=[0, 1])
        # fig.update_yaxes(range=[0, 1])
        # fig.add_layout_image(
        #     source=source,
        #     xref="x",
        #     yref="y",
        #     x=x0,
        #     y=y0,
        #     xanchor="left",
        #     yanchor="bottom",
        #     layer="below",
        #     sizing="stretch",
        #     sizex=xsize,
        #     sizey=ysize,
        # )
        return fig
    except:
        fig = px.scatter(width=500, height=500)
        fig.update_xaxes(range=[-10, 10])
        fig.update_yaxes(range=[-10, 10])
        return fig

if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')