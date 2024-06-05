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
import os
import time
import pickle

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
    dcc.Store(id='local_memory', storage_type='local'),
    html.Button('Button', id='button', n_clicks=0),
    dcc.Checklist(id='checklist', options=[{'label': 'Option 1', 'value': '1'}, {'label': 'Option 2', 'value': '2'}], value=['1']),
    html.H4('Experiment id'),
    dcc.Input(id='experiment-id', type='number', value=1, required=True), 
    html.P(id='max-experiment-id', children=''),
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

@app.callback(Output('max-experiment-id', 'children'),
              Input('button', 'n_clicks'),
                # prevent_initial_call=True,
                )
def get_max_experiment_id(n_clicks):
    db_gen = get_db()
    db = next(db_gen)
    query = db.query(Experiment.experiment_id).order_by(Experiment.experiment_id.desc()).first()
    return f"Max experiment id: {query[0]}"

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
    # print("*"*20 + "THE COMPUTE IS RUNNING" + "*"*20)
    # db_gen = get_db()
    # db = next(db_gen)
    # query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
    # df = pd.read_sql(query.statement, query.session.bind)
    # positions = pd.json_normalize(df['positions'])
    # x = positions['motors::X']
    # y = positions['motors::Y']

    # query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == report_name).order_by(Report.report_id.desc()).all()
    # for i in range(len(query)):
    #     img = np.asarray(query[i][0], dtype=np.uint8)
    #     n_measured = query[i][1]
    #     print(f"Image {i} shape: {img.shape} and n_measured: {n_measured}. Min: {img.min()} Max: {img.max()}")
    #     plt.imshow(img, origin='lower', extent=[x.min()-0.6, x.max(), y.min()-0.6, y.max()])
    #     plt.scatter(x[:n_measured], y[:n_measured], c='k', s=1, alpha=0.5)
    #     plt.axis('off')
    #     plt.savefig(f'save/imgs/{experiment_id}_{report_name}_{n_measured}_{i}.png', bbox_inches='tight')
    #     plt.clf()
        
    # print(len(query))

    # print("*"*20 + "THE COMPUTE IS DONE" + "*"*20)
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

@app.callback(Output('image', 'figure', allow_duplicate=True,),
              Input("scatter-plot", "clickData"),
            #   Input('image', 'figure'),
              Input('experiment-id', 'value'),
              prevent_initial_call=True,
            #   Input("scatter-plot", "hoverData"),
            # prevent_initial_call=True,
            )
def update_figure(clickData, experiment_id):
    start = time.perf_counter_ns()
    # print(f"{clickData=}")
    if clickData is None:
        return no_update
    measurement_id = clickData['points'][0]['customdata'][0]
    data = get_spectra(experiment_id, [measurement_id])[0]
    # all_ids = get_all_measurement_ids(experiment_id)
    # data = get_average_spectrum(experiment_id, all_ids)
    # data = resize(data, (128,128), anti_aliasing=True)
    # if not figure:
    figure = px.imshow(data, origin='lower', color_continuous_scale='Blues', aspect='auto')
    # else:
    #     figure['data'][0]['z'] = data
    print(f"Dash took {(time.perf_counter_ns()-start)/1e6:.02f}ms to update image.")
    return figure

@app.callback(
        Output('local_memory', 'data'),
        Output('image', 'figure', allow_duplicate=True,),
            #  Output('scatter-plot', 'figure', allow_duplicate=True,), 
            #  Input('scatter-plot', 'figure'),
            #   Input('report', 'figure'),
              Input('report', 'clickData'),
              Input('experiment-id', 'value'),
              prevent_initial_call=True,
)
def report_clicked(clickData, experiment_id):
    if clickData is None:
        return no_update
    print(clickData)
    db_gen = get_db()
    db = next(db_gen)
    x_i, y_i = clickData['points'][0]['x'], clickData['points'][0]['y']
    print(f"Clicked on x: {x_i}, y: {y_i}")
    query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == "UMAP interpolated coords").order_by(Report.report_id.desc()).first()
    image = np.asarray(query[0])
    umap_coords = image[y_i, x_i]
    print(f"UMAP coords of clicked spot: {umap_coords}")
    start = time.perf_counter()
    query = db.query(Report.data['umap_coords'], Report.data['data_ids'], Report.data['xy_coords']).filter(Report.experiment_id == experiment_id, Report.name == "umap_coords").order_by(Report.report_id.desc()).first()
    all_data_umap_coords = np.asarray(query[0])
    data_ids = query[1]
    xy_coords = np.asarray(query[2])
    n_closest = 25
    closest_indices = get_n_closest_spectra_indices(n_closest, umap_coords, all_data_umap_coords)
    spectrum = get_average_spectrum(experiment_id, [data_ids[i] for i in closest_indices])
    image_figure = px.imshow(spectrum, origin='lower', color_continuous_scale='Blues', aspect='auto')
    print(f"Dash took {(time.perf_counter()-start)*1000:.02f}ms to get and display average spectrum.")
    return {'integrated_xy_points': xy_coords[closest_indices]}, image_figure

def get_spectra(experiment_id, data_ids):
    db_gen = get_db()
    db = next(db_gen)
    query = db.query(Experiment.data_names_to_learn).filter(Experiment.experiment_id == experiment_id).first()
    data_fieldname = list(json.loads(query[0]).keys())[0]
    # measurement_ids=measurement_ids
    query = db.query(Data.data, Data.data_info).filter(Data.data_id.in_(data_ids)).all()#.order_by(Data.measurement_id.desc()).all()
    if query is None:
        raise ValueError(f"Data not found for data_ids: {data_ids}")
    data_filepath, data_info = query[0]
    data_info = json.loads(data_info)
    num_bytes = os.path.getsize(data_filepath)
    shape = data_info['dimensions']
    if np.prod(shape) * 4 == num_bytes:
        data = np.fromfile(data_filepath, dtype=np.int32).reshape(*shape, order='F')
    else:
        data = np.fromfile(data_filepath, dtype=np.float64).reshape(*shape, order='F')
    spectra = np.zeros((len(data_ids), *data.shape), dtype=data.dtype)
    spectra[0] = data
    print(f"{len(data_ids)=}, {len(set(data_ids))=}, {len(query)=}")
    if len(data_ids) == 1:
        return spectra
    for i, (data_filepath, data_info) in enumerate(query[1:len(data_ids)]):
        data_info = json.loads(data_info)
        spectra[i] = np.fromfile(data_filepath, dtype=spectra.dtype).reshape(*shape, order='F')
    return spectra

def get_n_closest_spectra_indices(n, umap_selection, all_data_umap_coords):
    distances = np.linalg.norm(all_data_umap_coords - umap_selection, axis=1)
    closest_indices = np.argsort(distances)[:n]
    print(f"Closest indices: {len(closest_indices)}")
    return closest_indices

def get_average_spectrum(experiment_id, measurement_ids):
    spectra = get_spectra(experiment_id, measurement_ids)
    return spectra.mean(axis=0)


@app.callback(Output('report', 'figure'),
              Input('button', 'n_clicks'),
              Input('experiment-id', 'value'),
              Input('report-type', 'value'),
              Input('interval-component', 'n_intervals'),
            #   Input("scatter-plot", "hoverData"),
            )
def update_report(n, experiment_id, report_type, _):
    start = time.perf_counter_ns()
    db_gen = get_db()
    db = next(db_gen)
    # query = db.query(Report).filter(Report.experiment_id == experiment_id).all()
    if report_type is None:
        query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id).order_by(Report.report_id.desc()).first()
    else:
        query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == report_type).order_by(Report.report_id.desc()).first()
    if query is None:
        print("Not updating figure")
        return no_update
    image = np.asarray(query[0])

    
    print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to query report for image EXPERIMENT ID {experiment_id}.")
    # print(query)
    

    # print(query)
    if len(image.shape) == 2:
        print('2D image')
        figure = px.imshow(image, origin='lower', color_continuous_scale='Viridis', title=f"Number of points: {query[1]}")
    elif len(image.shape) == 3:
        print('3D image')
        # image = (255 * (image - image.min(axis=0)[np.newaxis, ...]) / (image.max(axis=0)[np.newaxis, ...] - image.min(axis=0)[np.newaxis, ...])).astype(np.uint8)
        figure = px.imshow(image, origin='lower', title=f"Number of points: {query[1]}")
    if report_type == "UMAP as RGB":
        query = db.query(Report.data['image'], Report.data['n_measured']).filter(Report.experiment_id == experiment_id, Report.name == "UMAP interpolated coords").order_by(Report.report_id.desc()).first()
        umap_coords = np.asarray(query[0])
        figure.update(data=[{
            "customdata": umap_coords,
            'hovertemplate':  "UMAP: %{customdata[0]:.2f} %{customdata[1]:.2f} %{customdata[2]:.2f}",
        }])
    # print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to get image.")
    # figure = px.imshow(image, origin='lower', title=f"Number of points: {query[1]}")
    # print(f"Took {(time.perf_counter_ns()-start)/1e6:.02f}ms to make figure.")
    # else:
    #     return no_update
    print(f"Dash took {(time.perf_counter_ns()-start)/1e6:.02f}ms to update report.")
    if not figure:
        return no_update
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
    Output('scatter-plot', "figure"), 
    Input('interval-component', 'n_intervals'),
    Input('experiment-id', 'value'),
    # Input('n-clusters', 'value'),
    State('local_memory', 'data'),
)
def update_scatter_plot(n, experiment_id, local_memory):
    # try:
    db_gen = get_db()
    db = next(db_gen)
    # experiment = db.query(Experiment).filter(Experiment.experiment_id == experiment_id).first()
    # experiment.n_clusters = n_clusters
    # db.commit()
    query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
    df = pd.read_sql(query.statement, query.session.bind)
    # print(df.columns)
    positions=pd.json_normalize(df['positions'])
    positions['color'] = 'black'

    integrated_xy_points = local_memory.get('integrated_xy_points', None)
    if integrated_xy_points is not None:
        integrated_xy_points = np.asarray(integrated_xy_points)
        other_df = pd.DataFrame({'motors::X':integrated_xy_points[:,0], 'motors::Y':integrated_xy_points[:,1]})
    positions.loc[positions['motors::X'].isin(other_df['motors::X']) & positions['motors::Y'].isin(other_df['motors::Y']), 'color'] = 'red'
    #     fig.add_trace(go.Scatter(x=integrated_xy_points[:,0], y=integrated_xy_points[:,1], mode='markers', marker=dict(color='red')))
    # print(integrated_xy_points)
    # if integrated_xy_points is not None:
    # df = pd.DataFrame({'x_data':x_list, 'y_data':y_list})
    # df = pd.concat([positions, df], axis=0)
    fig = px.scatter(
        positions, x="motors::X", y="motors::Y", 
        width=500, height=500,
        color='color',
        hover_data = {
            'measurement_id' : df['measurement_id'],
        },
    )
    # fig.add = px.scatter(df, x='x_data', y='y_data').update_traces(marker=dict(color='red'))
    # fig = go.Figure(data = fig+fig2)
    #     fig = go.Figure(data = fig+fig2)
    #     fig.add_trace(go.Scatter(x=integrated_xy_points[:,0], y=integrated_xy_points[:,1], mode='markers', marker=dict(color='red')))
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
    # except Exception as e:
    #     print(f"Encountered an error here:\n{e}")
    #     fig = px.scatter(width=500, height=500)
    #     fig.update_xaxes(range=[-10, 10])
    #     fig.update_yaxes(range=[-10, 10])
    #     return fig

if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')