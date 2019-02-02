import plotly.plotly as py
import plotly.figure_factory as ff
import plotly 
import numpy as np

#plotly.tools.set_credentials_file(username='victorspruela', api_key='QsriLXOL5rTdL3vnzeda')

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

output_dir ='../Output'
features_raw = pd.read_csv(output_dir + '/all_features.csv', sep=";")
features  = features_raw.drop(['Unnamed: 0', 'signal_id', 'id_measurement', 'phase'], axis = 1).copy()
scaler = MinMaxScaler()
features_norm = pd.DataFrame(scaler.fit_transform(features), columns = features.columns)

scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features), columns = features.columns)
features_log = features.apply(np.log)

app = dash.Dash()

app.layout = html.Div(children=[
    
    html.H1(
        children='Data Visualization Dashboard',
        style={
            'textAlign': 'center'
        }
    ),

    html.Div([

        html.Div([
            html.H3('Configuration'),
            dcc.Dropdown(
                id='dropdown-features',
                options=[{'label': i, 'value': i} for i in features.columns],
                value=['number_of_peaks_p1','number_of_peaks_p2'],
                multi=True
            ),
        ], className="three columns"),

        html.Div([
            
            html.Div([
                html.Div([
                    html.H3('Line Plot'),
                    dcc.Graph(id='graph-line')
                ], className="six columns"),

                html.Div([
                    html.H3('Distribution Plot'),
                    dcc.Graph(id='graph')  
                ], className="six columns")

            ], className="row"),

            html.Div([
                html.H3('Pair Plot'),
                dcc.Graph(id='graph-pairs')  
            ], className="row")

           
        ], className="nine columns", style={'overflowY': 'scroll', 'height': 500})
    



    ], className="row"),


    
     



   
])

@app.callback(Output('graph', 'figure'), 
[Input('dropdown-features', 'value')])
def display_histgraphs(selected_values):
    traces = []    
    [traces.append(go.Histogram(x = (features_log.loc[:,str(val)].values),histnorm='probability', name = str(val))) for val in selected_values]
    
    layout = go.Layout(barmode='overlay', legend=dict(orientation="h"))

    return go.Figure(data = traces, layout = layout)

@app.callback(Output('graph-line', 'figure'), 
[Input('dropdown-features', 'value')])
def display_linegraphs(selected_values):
    traces = []
    [traces.append(go.Line(y = (features_log.loc[:,str(val)].values), name = str(val))) for val in selected_values]
    layout = go.Layout(legend=dict(orientation="h"))
    return go.Figure(data = traces, layout=layout)

@app.callback(Output('graph-pairs', 'figure'), 
[Input('dropdown-features', 'value')])
def display_pairplotgraphs(selected_values):
    data_dict = []
    for val in selected_values:
        data_dict.append(dict(label = val, values = features_log.loc[:,str(val)].values))


    color_vals = features_norm['target']
    pl_colorscale=[[0.0, 'red'],
               [1, 'blue']]
    trace = (go.Splom(dimensions= data_dict, 
                        marker=dict(color=color_vals,
                            size=7,
                            colorscale=pl_colorscale,
                            showscale=False,
                            )
            ))

    trace['diagonal'].update(visible=False)
    trace['showupperhalf']=False

    layout = go.Layout(
        height=600,
        autosize=True,
        hovermode='closest',
    )   


    return go.Figure(data = [trace], layout=layout)


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)