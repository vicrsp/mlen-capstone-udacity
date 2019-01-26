import plotly.plotly as py
import plotly.figure_factory as ff
import plotly 
import numpy as np

plotly.tools.set_credentials_file(username='victorspruela', api_key='QsriLXOL5rTdL3vnzeda')

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.preprocessing import MinMaxScaler
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

output_dir ='./Output'
features_raw = pd.read_csv(output_dir + '/all_features.csv', sep=";")
features  = features_raw.drop(['Unnamed: 0', 'signal_id', 'id_measurement', 'phase'], axis = 1).copy()
scaler = MinMaxScaler()
features_norm = pd.DataFrame(scaler.fit_transform(features), columns = features.columns)

hist_data = []

for column in features.columns[1:5]:
    hist_data.append(features.loc[:,column])

group_labels = features.columns[1:5]
    
# Create distplot with custom bin_size
# fig = ff.create_distplot(hist_data, group_labels)

app = dash.Dash()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(children=[
    
    html.H1(
        children='Feature Distribution Dashboard',
        style={
            'textAlign': 'center'
            #'color': colors['text']
        }
    ),
    
    dcc.Dropdown(
        id='dropdown-features',
        options=[{'label': i, 'value': i} for i in features.columns],
        value=['number_of_peaks_p1','number_of_peaks_p2'],
        multi=True
    ),
    dcc.Graph(
        id='graph'
    )
])

@app.callback(Output('graph', 'figure'), 
[Input('dropdown-features', 'value')])
def display_graphs(selected_values):
    traces = []
    print(selected_values)
    
    [traces.append(go.Histogram(x = (features.loc[:,str(val)].values),histnorm='probability', name = str(val))) for val in selected_values]
    
    layout = go.Layout(barmode='overlay')

    return go.Figure(data = traces, layout = layout)
        

if __name__ == '__main__':
    app.run_server(debug=True)