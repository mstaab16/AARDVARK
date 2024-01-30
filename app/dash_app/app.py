import dash
# import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly.express as px

import pandas as pd
import numpy as np
import json
import base64

from db.database import get_db
from db.models import Experiment, Decision, Measurement, Data, Report
from maestro_api import maestro_messages as mm

app = dash.Dash(__name__,#  external_stylesheets=[dbc.themes.BOOTSTRAP]
                ### THE ONLY CHANGE FROM THE ABOVE, PLEASE ADD IN THIS LINE
                requests_pathname_prefix='/dashboard/')


app.layout = html.Div([
    html.H4('Interactive scatter plot'),
    dcc.Input(id='experiment-id', type='number', value=1, required=True),
    dcc.Graph(id="scatter-plot"),
    dcc.Graph(id="image"),
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
)
def update_bar_chart(n, experiment_id):
    try:
        db_gen = get_db()
        db = next(db_gen)
        query = db.query(Measurement).filter(Measurement.experiment_id == experiment_id, Measurement.measured == True).order_by(Measurement.ai_cycle)
        df = pd.read_sql(query.statement, query.session.bind)
        positions=pd.json_normalize(df['positions'])
        fig = px.scatter(
            positions, x="motors::X", y="motors::Y", 
            width=500, height=500,
            # color="species", size='petal_length', 
            # hover_data=['petal_width'])
        )
        fig.update_xaxes(range=[-10, 10])
        fig.update_yaxes(range=[-10, 10])
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
        data = db.query(Data).filter(Data.experiment_id == experiment_id, Data.fieldname == "Fixed_Spectra0").order_by(Data.measurement_id.desc()).first().data
        # df = pd.read_sql(query.statement, query.session.bind)
        data = mm.FitsDescriptor.model_validate_json(json.loads(data)).Data
        data =  base64.decodebytes(data)
        data = np.frombuffer(data, dtype=np.int32).reshape((64,64))
        fig = px.imshow(data)
    except Exception as e:
        print(e)
        # pass
        fig = px.imshow(np.random.uniform(0,1,(64,64)), width=500, height=500)
        # fig.update_xaxes(range=[-10, 10])
        # fig.update_yaxes(range=[-10, 10])
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)