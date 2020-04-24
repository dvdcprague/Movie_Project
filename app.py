import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib
from ast import literal_eval
import json
import re


from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MultiLabelBinarizer

from scipy.stats import mannwhitneyu
from lightgbm import LGBMRegressor
#from pdpbox import pdp, get_dataset, info_plots
import pdp
from flask_caching import Cache


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60

server = app.server
app.config.suppress_callback_exceptions = True


# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data
data = pd.read_csv(DATA_PATH.joinpath("feature_analysis_data.csv"))

# read package-feature mapping
feature_mapping = pd.read_csv(DATA_PATH.joinpath('Package_Feature_Mapping.csv'))

# read feature code feature name mapping
feature_list = pd.read_csv(DATA_PATH.joinpath('feature_list.csv'))
feature_list_dict = dict(zip(feature_list['feature code'], feature_list['feature name']))

# read epi jdp feature mapping
epi_jdp_feature_mapping = pd.read_csv(DATA_PATH.joinpath('epi_jdp_feature_mapping.csv'))


# Data processing
data['features'] = data['features'].apply(literal_eval)
data['packages'] = data['packages'].apply(literal_eval)
data['options'] = data['options'].apply(literal_eval)

#data['sales_year_month'] = data['transaction_year'].map(int).map(str) + '-' + data['transaction_month'].map(int).map(str)
#data['sales_year_month'] = pd.to_datetime(data['sales_year_month'])
#data['sales_quarter'] = data['sales_year_month'].dt.to_period("Q")
#data['exterior_color'] = data['exterior_color'].fillna('Unknown')
#
## generate sales volume dataframe for each trim, region, and sales quarter for future use
#sales_volume = data.groupby(['trim', 'sales_quarter', 'region'])['vin'].count().reset_index()
#
#data.loc[data['packages'].isnull(),['packages']] = data.loc[data['packages'].isnull(),'packages'].apply(lambda x: [])
#data.loc[data['options'].isnull(),['options']] = data.loc[data['options'].isnull(),'options'].apply(lambda x: [])
#
#data['packages'] = data['packages'].apply(lambda x: x.replace('[', '').replace(']', '').split(',') if x != [] else x)
#data['options'] = data['options'].apply(lambda x: x.replace('[', '').replace(']', '').split(',') if x != [] else x)
#
#data['packages'] = data['packages'].apply(lambda x: [p.strip() for p in x])
#data['options'] = data['options'].apply(lambda x: [o.strip() for o in x])
#
#
#data['features'] = data['features'].apply(literal_eval)

# Remove accessories from package/option list
#def remove(l, item):
#    if item in l:
#        l.remove(item)
#    return l
#
#data['packages'] = data['packages'].apply(lambda x: remove(x, 'Int. Lighting Pkg.'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Illuminated Kick Plates'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Midnight Edition Floor Mats Plus Trunk Mat (5 Piece)'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Splash Guards'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Sport Floor Mats Plus Trunk Mat (5 Piece)'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Chrome Bumper Protector'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Rear Spoiler'))
#data['options'] = data['options'].apply(lambda x: remove(x, 'Led Fog Lights'))


# generate drop-down list for make, model, modelyear, trim, region, and time of the year
make_list = data['make'].unique().tolist()
model_list = data['model'].unique().tolist()
modelyear_list = data['modelyear'].unique().tolist()
trim_list = data['trim'].unique().tolist()
region_list = data['region'].unique().tolist()
#region_list = [x for x in region_list if pd.notnull(x)]
region_list = ['National'] + region_list
data['sales_quarter'] = data['sales_quarter'].apply(str)
time_list  = sorted(data['sales_quarter'].unique().tolist())
#time_list.remove('nan')
color_list = ['All Colors'] + data['exterior_color'].unique().tolist()

####################################


color_legend = {'green': 'Positively Impact DTT', 'red': 'Negatively Impact DTT', 'black': 'Minor Impact',
                'grey': 'Not Significant'}
effect_color = {'Positively Impact DTT': 'green', 'Negatively Impact DTT': 'red',
                'Not Significant': 'grey'}

def color(row, non_sig):
    if row['Feature'] in non_sig:
        return 'grey'
    else:
        if row['Value'] < 0:
            return 'green'
        else:
            return 'red'   

def color_comparator(row, non_sig):
    if row['package_option'] in non_sig:
        return 'grey'
    else:
        if row['value'] < 0:
            return 'green'
        else:
            return 'red'   


### word spliting function
def splitTextonWords(Text, numberOfWords=1):
    if (numberOfWords > 1):
        text = Text.lstrip()
        pattern = '(?:\S+\s*){1,'+str(numberOfWords-1)+'}\S+(?!=\s*)'
        x =re.findall(pattern,text)
    elif (numberOfWords == 1):
        x = Text.split()
    else: 
        x = None
    return '<br>'.join(x)
############################################

# generate filtered dataset
def ohe_features(df, col):
    mlb = MultiLabelBinarizer()
    d = pd.DataFrame(
        mlb.fit_transform(df[col].values)
        , df.index, mlb.classes_
    )
    return df.drop(col, 1).join(d)


def get_sample_data(make, model, modelyear, trim, color, region, sales_quarter):
#     df = ohe_features(df, 'features')
#     df.dropna(subset=[target], inplace=True)
    df = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear) & (data['trim'] == trim)]
    
    if color != 'All Colors':
        df = df[df['exterior_color'] == color]
        
    if region != 'National':
        df = df[df['region'] == region]
    
    df = df[df["sales_quarter"].isin(sales_quarter)]
    
    result = ohe_features(df, 'packages')
    result = ohe_features(result, 'options')
    
    return result

def effect(color):
    if color == 'green':
        return 'Positively Impact DTT'
    elif color == 'red':
        return 'Negatively Impact DTT'
    else:
        return 'Not Significant'

def common_elements(list1, list2):
    result = [element for element in list1 if element in list2]
    return len(result)

def mannwhitneyu_test(c,data1,data2):

    # compare samples
    stat, p = mannwhitneyu(data1, data2)#wilcoxon(data1, data2)
#     print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        output = ('Same Mean (fail to reject H0)')
    else:
        output = ('Different Mean (reject H0)')
        
    return [c, stat, p]

### LGBM parameters for different make, model, trims
def lgbm_param(trim):
    if trim == '3.0 40i':
        learning_rate = 0.2
        max_depth = 15
        n_estimators = 600
    if trim == '4.4 50i':
        learning_rate = 0.1
        max_depth = 20
        n_estimators = 200
    if trim == '2.0 Premium':
        learning_rate = 0.2
        max_depth = 8
        n_estimators = 400
    if trim == '2.0 Premium Plus':
        learning_rate = 0.5
        max_depth = 8
        n_estimators = 600
    if trim == '3.0 Premium':
        learning_rate = 0.4
        max_depth = 8
        n_estimators = 400
    if trim == '3.0 Premium Plus':
        learning_rate = 0.4
        max_depth = 8
        n_estimators = 400
    if trim == '3.0 Prestige':
        learning_rate = 0.2
        max_depth = 12
        n_estimators = 600
    if trim == 'GLE400 3.0L I6':
        learning_rate = 0.4
        max_depth = 12
        n_estimators = 600
    if trim == 'AMG GLE63 S Coupe 5.5L V8':
        learning_rate = 0.01
        max_depth = 8
        n_estimators = 400
    if trim == 'GLE43 AMG Coupe 3.0L V6':
        learning_rate = 0.1
        max_depth = 15
        n_estimators = 600
    if trim == 'AMG GLE43 3.0L V6':
        learning_rate = 0.05
        max_depth = 8
        n_estimators = 400
    return learning_rate, max_depth, n_estimators
    


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
#            html.H5("Days to Turn Analytics"),
            html.H3("Days to Turn Analytics Demo: Curated Dataset"),
#            html.Div(
#                id="intro",
#                children="Days to Turn analytics demo, 2018 Nissan Altima",
#            ),
        ],
    )

def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Make"),
            dcc.Dropdown(
                id="make-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in make_list],
                value=make_list[0],
            ),
            html.Br(),
            html.P("Select Model"),
            dcc.Dropdown(
                id="model-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in model_list],
                value=model_list[0],
            ),
            html.Br(),
            html.P("Select Model Year"),
            dcc.Dropdown(
                id="modelyear-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in modelyear_list],
                value=modelyear_list[0],
            ),
            html.Br(),
            html.P("Select Trim"),
            dcc.Dropdown(
                id="trim-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in trim_list],
                value=trim_list[0],
            ),
            html.Br(),
            html.P("Select Exterior Color"),
            dcc.Dropdown(
                id="color-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in color_list],
                value=color_list[0],
#                multi = True,
            ),
            html.Br(),
            html.P("Select Region"),
            dcc.Dropdown(
                id="region-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in region_list],
                value=region_list[0],
#                multi = True,
            ),
            html.Br(),
            html.P("Select Time of the Year"),
            dcc.Dropdown(
                id="time-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in time_list],
                value=time_list[:],
                multi = True,
            ),
            html.Br(),
            html.Br(),

            html.Div(
                id="button",
                children=[html.Button('Submit', id='submit-button', n_clicks=0), html.Button('Reset', id="reset-btn", n_clicks=0)],
#                children=[
#            html.Div(
#                id='submit-button-container',
#                className="two columns",
#                children=html.Button(id='submit-button', n_clicks=0, children='Submit'),
#            ),
#            html.Div(
#                id='reset-button-container',
#                className="two columns",
#                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
#            ),]
            ),
            
        ],
    )

def comparator1_description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="comparator1-description-card",
        children=[
#            html.H5("Days to Turn Analytics"),
            html.H5("Please Select Model #1: "),
#            html.Div(
#                id="intro",
#                children="Days to Turn analytics demo, 2018 Nissan Altima",
#            ),
        ],
    )

def comparator1_generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="comparator1-control-card",
        children=[
            html.P("Select Make"),
            dcc.Dropdown(
                id="comparator1-make-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in make_list],
                value=make_list[0],
            ),
            html.Br(),
            html.P("Select Model"),
            dcc.Dropdown(
                id="comparator1-model-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in model_list],
                value=model_list[0],
            ),
            html.Br(),
            html.P("Select Model Year"),
            dcc.Dropdown(
                id="comparator1-modelyear-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in modelyear_list],
                value=modelyear_list[0],
            ),
            html.Br(),
            html.P("Select Trim"),
            dcc.Dropdown(
                id="comparator1-trim-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in trim_list],
                value=trim_list[0],
            ),
#                children=[
#            html.Div(
#                id='submit-button-container',
#                className="two columns",
#                children=html.Button(id='submit-button', n_clicks=0, children='Submit'),
#            ),
#            html.Div(
#                id='reset-button-container',
#                className="two columns",
#                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
#            ),]

            
        ],
    )


#def comparator1_selected_card(comparator1make, comparator1model, comparator1modelyear, comparator1trim):
#    """
#
#    :return: A Div containing dashboard title & descriptions.
#    """
#    
#    selected = str(comparator1modelyear) + ' ' + comparator1make + ' ' + comparator1model + ': ' + comparator1trim
#    
#    return html.Div(
#        id="comparator1-selected-card",
#        children=[
##            html.H5("Days to Turn Analytics"),
#            html.H5("Selected Model #1: ", style={'height': '30px', 'width': '400px'}),
#            html.H6(selected),
#        ],
#    )


def comparator2_description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="comparator2-description-card",
        children=[
            html.Br(),
            html.Br(),
            html.H5("Please Select Model #2: "),
#            html.Div(
#                id="intro",
#                children="Days to Turn analytics demo, 2018 Nissan Altima",
#            ),
        ],
    )


def comparator2_generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="comparator2-control-card",
        children=[
       
            html.P("Select Make"),
            dcc.Dropdown(
                id="comparator2-make-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in make_list],
                value=make_list[0],
            ),
            html.Br(),
            html.P("Select Model"),
            dcc.Dropdown(
                id="comparator2-model-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in model_list],
                value=model_list[0],
            ),
            html.Br(),
            html.P("Select Model Year"),
            dcc.Dropdown(
                id="comparator2-modelyear-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in modelyear_list],
                value=modelyear_list[0],
            ),
            html.Br(),
            html.P("Select Trim"),
            dcc.Dropdown(
                id="comparator2-trim-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in trim_list],
                value=trim_list[0],
            ),
            html.Div(
                id="button",                
                children=[html.Br(), html.Button('Submit', id='comparator-submit-button', n_clicks=0), 
                          dcc.Link(html.Button('Main Page'),href='/dash-dtt-analytics')],

            ),
            
        ],
    )

#def comparator2_selected_card(comparator2make, comparator2model, comparator2modelyear, comparator2trim):
#    """
#
#    :return: A Div containing dashboard title & descriptions.
#    """
#    selected = str(comparator2modelyear) + ' ' + comparator2make + ' ' + comparator2model + ': ' + comparator2trim
#    
#    return html.Div(
#        id="comparator2-selected-card",
#        children=[
##            html.H5("Days to Turn Analytics"),
#            html.Br(),
#            html.Br(),
#            html.H5("Selected Model #2: ", style={'height': '30px', 'width': '400px'}),
#            html.H6(selected),
##            html.Div(
##                id="button",                
##                children=[html.Br(), html.Button('Submit', id='comparator-trim-submit-button', n_clicks=0)],
##
##            ),
#        ],
#    )


## function to calculate key performance indicators for comparator overview section
def comparator_overview(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):
    comparator1_df = data[(data['make'] == make1) & (data['model'] == model1) & (data['modelyear'] == modelyear1) & (data['trim'] == trim1)]
    comparator2_df = data[(data['make'] == make2) & (data['model'] == model2) & (data['modelyear'] == modelyear2) & (data['trim'] == trim2)]
    
    avg_msrp_1 = round(comparator1_df['msrp'].mean(), 1)
    avg_dtt_1 = round(comparator1_df['days_to_turn'].mean(), 1)
    avg_gross_profit_1 = round(comparator1_df['veh_gross_profit'].mean(), 1)
    avg_incentives_1 = round(comparator1_df['incentives'].mean(), 1)
    avg_cftp_1 = round(comparator1_df['cftp'].mean(), 1)
    
    avg_msrp_2 = round(comparator2_df['msrp'].mean(), 1)
    avg_dtt_2 = round(comparator2_df['days_to_turn'].mean(), 1)
    avg_gross_profit_2 = round(comparator2_df['veh_gross_profit'].mean(), 1)
    avg_incentives_2 = round(comparator2_df['incentives'].mean(), 1)
    avg_cftp_2 = round(comparator2_df['cftp'].mean(), 1)
    
    result = {'Average MSRP': [avg_msrp_1, avg_msrp_2], 'Average Days to Turn': [avg_dtt_1, avg_dtt_2], 
         'Average Gross Profit': [avg_gross_profit_1, avg_gross_profit_2], 'Average Incentives': [avg_incentives_1, avg_incentives_2],
         'Average CFTP': [avg_cftp_1, avg_cftp_2]}
    
    return result


## function to generate comparator standard features dataframe
def comparator_standard_features(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):
    comparator1_df = data[(data['make'] == make1) & (data['model'] == model1) & (data['modelyear'] == modelyear1) & (data['trim'] == trim1)]
    comparator2_df = data[(data['make'] == make2) & (data['model'] == model2) & (data['modelyear'] == modelyear2) & (data['trim'] == trim2)]
    
    comparator1_df_result = comparator1_df.groupby('trim')['features'].apply(list).reset_index(name='features_list')
    comparator1_df_result['common_features'] = comparator1_df_result['features_list'].apply(lambda x: set.intersection(*map(set, x)))
    comparator1_df_result['common_features'] = comparator1_df_result['common_features'].apply(lambda x: sorted(list(x)))
    comparator1_df_result.drop('features_list', axis= 1, inplace = True)
    comparator1_df_final = pd.merge(comparator1_df, comparator1_df_result, on = 'trim', how = 'left')
    comparator1_df_final['common_features_names'] = comparator1_df_final['common_features'].apply(lambda x: [feature_list_dict[f] for f in x if f != 99999999])
    comparator1_common_features = comparator1_df_final['common_features_names'].iloc[0]
    
    comparator2_df_result = comparator2_df.groupby('trim')['features'].apply(list).reset_index(name='features_list')
    comparator2_df_result['common_features'] = comparator2_df_result['features_list'].apply(lambda x: set.intersection(*map(set, x)))
    comparator2_df_result['common_features'] = comparator2_df_result['common_features'].apply(lambda x: sorted(list(x)))
    comparator2_df_result.drop('features_list', axis= 1, inplace = True)
    comparator2_df_final = pd.merge(comparator2_df, comparator2_df_result, on = 'trim', how = 'left')
    comparator2_df_final['common_features_names'] = comparator2_df_final['common_features'].apply(lambda x: [feature_list_dict[f] for f in x if f != 99999999])
    comparator2_common_features = comparator2_df_final['common_features_names'].iloc[0]
    
    comparator1_cf = pd.DataFrame({'feature_name': comparator1_common_features})
    comparator2_cf = pd.DataFrame({'feature_name': comparator2_common_features})
    
    comparator1_cf = pd.merge(comparator1_cf, epi_jdp_feature_mapping, left_on = 'feature_name', right_on = 'jdp_feature', how='left')
    comparator1_cf.drop_duplicates(inplace=True)
    
    comparator2_cf = pd.merge(comparator2_cf, epi_jdp_feature_mapping, left_on = 'feature_name', right_on = 'jdp_feature', how='left')
    comparator2_cf.drop_duplicates(inplace=True)
    
    comparator1_cf.drop(['jdp_feature_code', 'jdp_feature'], axis=1, inplace=True)
    comparator2_cf.drop(['jdp_feature_code', 'jdp_feature'], axis=1, inplace=True)
    
    comparator1_cf['comparator1'] = 'yes'
    comparator2_cf['comparator2'] = 'yes'
    
    comparator_cf = pd.merge(comparator1_cf, comparator2_cf, on = ['feature_name', 'group'], how = 'outer')
    comparator_cf.sort_values(by='group', inplace=True)
    comparator_cf.fillna('no', inplace=True)
    comparator_cf = comparator_cf[['group', 'feature_name', 'comparator1', 'comparator2']]
    
    return len(comparator1_common_features), len(comparator2_common_features), comparator_cf
    
def comparator_non_standard_features_take_rate(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):
    comparator1_df = data[(data['make'] == make1) & (data['model'] == model1) & (data['modelyear'] == modelyear1) & (data['trim'] == trim1)]
    comparator2_df = data[(data['make'] == make2) & (data['model'] == model2) & (data['modelyear'] == modelyear2) & (data['trim'] == trim2)]

    comparator1_df_result = comparator1_df.groupby('trim')['features'].apply(list).reset_index(name='features_list')
    comparator1_df_result['common_features'] = comparator1_df_result['features_list'].apply(lambda x: set.intersection(*map(set, x)))
    comparator1_df_result['common_features'] = comparator1_df_result['common_features'].apply(lambda x: sorted(list(x)))
    comparator1_df_result.drop('features_list', axis= 1, inplace = True)
    comparator1_df_final = pd.merge(comparator1_df, comparator1_df_result, on = 'trim', how = 'left')
    comparator1_df_final['unique_features'] = comparator1_df_final.apply(lambda row: sorted(list(set(row['features']) - set(row['common_features']))), axis=1)
    comparator1_df_final['unique_features_names'] = comparator1_df_final['unique_features'].apply(lambda x: [feature_list_dict[f] for f in x if f != 99999999])
    comparator1_df_final_ohe = ohe_features(comparator1_df_final, 'unique_features_names')
    comparator1_df_final_ohe.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain',
           'veh_fuel_type', 'liters', 'cylinders', 'doors', 'exterior_color',
           'msrp', 'region', 'division', 'packages', 'options', 'features',
           'days_to_turn', 'veh_gross_profit', 'incentives', 'cftp',
           'transaction_year', 'transaction_month', 'sales_year_month',
           'sales_quarter', 'common_features', 'unique_features'], axis=1, inplace=True)
    comparator1_non_standard_features = comparator1_df_final_ohe.columns.tolist()
    comparator1_non_standard_features_tr = []
    for nsf in comparator1_non_standard_features:
        take_rate = 100.0 * comparator1_df_final_ohe[comparator1_df_final_ohe[nsf] == 1].shape[0] / comparator1_df_final_ohe.shape[0]
        take_rate = round(take_rate, 1)
        take_rate_string = str(take_rate) + '%'
        comparator1_non_standard_features_tr.append(take_rate_string)
    comparator1_nsf = pd.DataFrame({'feature_name': comparator1_non_standard_features, 'comparator1_take_rate': comparator1_non_standard_features_tr})
    
    comparator2_df_result = comparator2_df.groupby('trim')['features'].apply(list).reset_index(name='features_list')
    comparator2_df_result['common_features'] = comparator2_df_result['features_list'].apply(lambda x: set.intersection(*map(set, x)))
    comparator2_df_result['common_features'] = comparator2_df_result['common_features'].apply(lambda x: sorted(list(x)))
    comparator2_df_result.drop('features_list', axis= 1, inplace = True)
    comparator2_df_final = pd.merge(comparator2_df, comparator2_df_result, on = 'trim', how = 'left')
    comparator2_df_final['unique_features'] = comparator2_df_final.apply(lambda row: sorted(list(set(row['features']) - set(row['common_features']))), axis=1)
    comparator2_df_final['unique_features_names'] = comparator2_df_final['unique_features'].apply(lambda x: [feature_list_dict[f] for f in x if f != 99999999])
    comparator2_df_final_ohe = ohe_features(comparator2_df_final, 'unique_features_names')
    comparator2_df_final_ohe.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain',
           'veh_fuel_type', 'liters', 'cylinders', 'doors', 'exterior_color',
           'msrp', 'region', 'division', 'packages', 'options', 'features',
           'days_to_turn', 'veh_gross_profit', 'incentives', 'cftp',
           'transaction_year', 'transaction_month', 'sales_year_month',
           'sales_quarter', 'common_features', 'unique_features'], axis=1, inplace=True)
    comparator2_non_standard_features = comparator2_df_final_ohe.columns.tolist()
    comparator2_non_standard_features_tr = []
    for nsf in comparator2_non_standard_features:
        take_rate = 100.0 * comparator2_df_final_ohe[comparator2_df_final_ohe[nsf] == 1].shape[0] / comparator2_df_final_ohe.shape[0]
        take_rate = round(take_rate, 1)
        take_rate_string = str(take_rate) + '%'
        comparator2_non_standard_features_tr.append(take_rate_string)
    comparator2_nsf = pd.DataFrame({'feature_name': comparator2_non_standard_features, 'comparator2_take_rate': comparator2_non_standard_features_tr})
    
    comparator1_nsf = pd.merge(comparator1_nsf, epi_jdp_feature_mapping, left_on = 'feature_name', right_on = 'jdp_feature', how='left')
    comparator1_nsf.drop_duplicates(inplace=True)

    comparator2_nsf = pd.merge(comparator2_nsf, epi_jdp_feature_mapping, left_on = 'feature_name', right_on = 'jdp_feature', how='left')
    comparator2_nsf.drop_duplicates(inplace=True)
    
    comparator1_nsf.drop(['jdp_feature_code', 'jdp_feature'], axis=1, inplace=True)
    comparator2_nsf.drop(['jdp_feature_code', 'jdp_feature'], axis=1, inplace=True)
    
    comparator_nsf = pd.merge(comparator1_nsf, comparator2_nsf, on = ['feature_name', 'group'], how = 'outer')
    comparator_nsf.sort_values(by='group', inplace=True)
    comparator_nsf.fillna('N/A', inplace=True)
    comparator_nsf = comparator_nsf[['group', 'feature_name', 'comparator1_take_rate', 'comparator2_take_rate']]
    
    return comparator_nsf



def generate_package_feature_imp(make, model, modelyear, trim, region, sales_quarter, bar_click, reset):
    
    sample_data = get_sample_data(make, model, modelyear, trim, region, sales_quarter)
    X = sample_data.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
 'exterior_color', 'msrp', 'region', 'division','days_to_turn', 'veh_gross_profit', 'incentives', 'cftp',
            'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter'], axis = 1)
    y = sample_data['days_to_turn']
    
    clf = BayesianRidge(compute_score=True)
    clf.fit(X, y)
    
    feature_imp = pd.DataFrame(zip(clf.coef_, X.columns), columns=['Value','Feature'])
    feature_imp = feature_imp.sort_values(by='Value')
    
    feature_imp['feature_type'] = feature_imp['Feature'].apply(lambda x: 'package' if isinstance(x, str) else 'feature')
    package_imp = feature_imp[feature_imp['feature_type'] == 'package']
    features_imp = feature_imp[feature_imp['feature_type'] == 'feature']
    package_imp.drop('feature_type', axis=1, inplace=True)
    features_imp.drop('feature_type', axis=1, inplace=True)
    package_imp.columns = ['value', 'package']
    features_imp.columns = ['value', 'feature']
    package_imp['value'] = package_imp['value'].apply(lambda x: round(x, 3))
    features_imp['value'] = features_imp['value'].apply(lambda x: round(x, 3))
    
    final_imp = pd.merge(features_imp, package_imp, on = 'value', how = 'left')
    final_imp['feature_name'] = final_imp['feature'].apply(lambda x: feature_list_dict[x])
    final_imp['plot_name'] = final_imp.apply(lambda row: row['package'] if pd.notnull(row['package']) else row['feature_name'], axis=1)
    final_imp['plot_code'] = final_imp.apply(lambda row: row['package'] if pd.notnull(row['package']) else row['feature'], axis=1)
    
    plot_data = pd.DataFrame({'plot_name': final_imp['plot_name'].unique().tolist(), 'plot_value': final_imp['value'].unique().tolist()})
    plot_data['color'] = plot_data['plot_value'].apply(lambda x: color(x))

    
    annotation = final_imp.groupby('plot_name')['feature_name'].apply(lambda x: ', '.join(x)).reset_index()
    
    hover_text = []
    for item in plot_data['plot_name']:
        text = 'Included Features: ' + '<br>' + annotation[annotation['plot_name'] == item].iloc[0]['feature_name']
        hover_text.append(text)
    
          

    data = [
        dict(
            x=plot_data['plot_value'],
            y=plot_data['plot_name'],
            type="bar",
            name="Feature Importance",
            hovertext= hover_text,
            marker=dict(
            color=plot_data['color'].tolist(),
            reversescale = True
        ),
        orientation='h',
        )
    ]

    layout = dict(
        title='Differential Packages/Features',
         width = 1000, height = 450,
        yaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=2, linecolor='black',
            showticklabels=True,
            automargin = True,mirror=True,
    #        tickfont = dict(color='green', family = "Arial", size = 13)
    #         domain=[0, 0.85],
        ),
        xaxis=dict(
            showline=True,
            linewidth=2, linecolor='black',
            zeroline=True, zerolinewidth=4, zerolinecolor='LightPink',mirror=True
    #         domain=[0, 0.85],
        ))

    
    return {"data": data, "layout": layout}


def generate_table_row(id, style, col1, col2, col3, col4, col5):
    """ Generate table rows.

    :param id: The ID of table row.
    :param style: Css style of this row.
    :param col1 (dict): Defining id and children for the first column.
    :param col2 (dict): Defining id and children for the second column.
    :param col3 (dict): Defining id and children for the third column.
    """

    return html.Div(
        id=id,
        className="row table-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                style={"textAlign": "center", "height": "150%"},
                className="two columns row-package",
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "height": "150%"},
                className="three columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"textAlign": "center", "height": "150%"},
                className="two columns",
                children=col3["children"],
            ),
            html.Div(
                id=col4["id"],
                style={"textAlign": "center", "height": "150%"},
                className="two columns",
                children=col4["children"],
            ),
            html.Div(
                id=col5["id"],
                style={"textAlign": "center", "height": "150%"},
                className="three columns",
                children=col5["children"],
            ),
#            html.Div(
#                id=col6["id"],
#                style={"textAlign": "center", "height": "150%"},
#                className="two columns",
#                children=col6["children"],
#            ),

        ],
    )

def generate_patient_table(target_package, included_features, sample_size, take_rate, dtt_change):
    """
    :param score_xrange: score plot xrange [min, max].
    :param wait_time_xrange: wait time plot xrange [min, max].
    :param figure_list:  A list of figures from current selected metrix.
    :param departments:  List of departments for making table.
    :return: Patient table.
    """
    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "30px"},
            {"id": "header_package", "children": html.B("Package/Option")},
            {"id": "header_include_features", "children": html.B("Included Features")},
            {"id": "header_sample_size", "children": html.B("Sample Size")},
            {"id": "header_take_rate", "children": html.B("Take Rate")},
            {"id": "header_dtt_change", "children": html.B("DTT Change with Package/Option")},
#            {"id": "header_w/o_ratio", "children": html.B("W/O Ratio")},
        )
    ]

    row = [
        generate_table_row(
            "package",
            {},
            {"id": "package", "children": html.P(target_package)},
            {"id": "include_features", "children": html.P(included_features)},
            {"id": "sample_size", "children": html.P(sample_size)},
            {"id": "take_rate", "children": html.P(take_rate)},
            {"id": "dtt_change", "children": html.P(dtt_change)},
#            {"id": "w/o_ratio", "children": html.P(round((dtt_w_package/dtt_wo_package), 2))},
        )
    ]
    
    header.extend(row)

    return header

def generate_simulator_row(id, style, col1, col2):
    """ Generate table rows.

    :param id: The ID of table row.
    :param style: Css style of this row.
    :param col1 (dict): Defining id and children for the first column.
    :param col2 (dict): Defining id and children for the second column.
    :param col3 (dict): Defining id and children for the third column.
    """

    return html.Div(
        id=id,
        className="row table-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                style={"display": "table", "height": "100%"},
                className="five columns row",
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "height": "100%", 'font-size': '26px'},
                className="five columns",
                children=col2["children"],
            ),
        ],
    )

def generate_simulator_table(diff_packages):
    """
    :param score_xrange: score plot xrange [min, max].
    :param wait_time_xrange: wait time plot xrange [min, max].
    :param figure_list:  A list of figures from current selected metrix.
    :param departments:  List of departments for making table.
    :return: Patient table.
    """
    # header_row
    header = [
        generate_simulator_row(
            "header",
            {"height": "50px", 'font-size': '26px'},
            {"id": "header_diff_package", "children": html.B("Differential Packages/Options")},
            {"id": "header_sim_dtt", "children": html.B("Simulated Days to Turn")},

        )
    ]

    row = [
        generate_simulator_row(
            "package",
            {},
            {"id": "diff_package", "children": dcc.Dropdown(
                id="diff-package-select",
                options=[{"label": i, "value": i} for i in diff_packages],
                value=diff_packages[0],
                multi = True,
            ),},
            {"id": "sim_dtt", "children": html.H3('')},

        )
    ]
    
    header.extend(row)

    return header

def initialize_table():
    """
    :return: empty table children. This is intialized for registering all figure ID at page load.
    """

    # header_row
    header = [
        generate_table_row(
            "header",
            {"height": "20px"},
            {"id": "header_package", "children": html.B("")},
            {"id": "header_include_features", "children": html.B("")},
            {"id": "header_sample_size", "children": html.B("")},
            {"id": "header_take_rate", "children": html.B("")},
            {"id": "header_dtt_change", "children": html.B("")},
#            {"id": "header_w/o_ratio", "children": html.B("")},
        )
    ]
    
    empty_table = header

    return empty_table

def initialize_simulator_table():
    """
    :return: empty table children. This is intialized for registering all figure ID at page load.
    """

    # header_row
    header = [
            generate_simulator_row(
                "header",
                {"height": "50px"},
                {"id": "header_diff_package", "children": html.B("")},
                {"id": "header_sim_dtt", "children": html.B("")},
    
            )
        ]


    empty_table = header

    return empty_table


def generate_comparator_overview_table_row(id, style, col1, col2, col3):
    """ Generate table rows.

    :param id: The ID of table row.
    :param style: Css style of this row.
    :param col1 (dict): Defining id and children for the first column.
    :param col2 (dict): Defining id and children for the second column.
    :param col3 (dict): Defining id and children for the third column.
    """

    return html.Div(
        id=id,
        className="row table-row",
        style=style,
        children=[
            html.Div(
                id=col1["id"],
                style={"textAlign": "center", "height": "100%"},
                className="four columns row-package",
                children=col1["children"],
            ),
            html.Div(
                id=col2["id"],
                style={"textAlign": "center", "height": "100%"},
                className="four columns",
                children=col2["children"],
            ),
            html.Div(
                id=col3["id"],
                style={"textAlign": "center", "height": "100%"},
                className="four columns",
                children=col3["children"],
            ),

        ],
    )

def initialize_comparator_overview_table():
    """
    :return: empty table children. This is intialized for registering all figure ID at page load.
    """

    # header_row
    header = [
        generate_comparator_overview_table_row(
            "header",
            {"height": "20px"},
            {"id": "header_category", "children": html.B("")},
            {"id": "header_model1", "children": html.B("")},
            {"id": "header_model2", "children": html.B("")},
        )
    ]
    
    empty_table = header
    return empty_table



mainPage = html.Div(
    id="app-container",
    children=[
#        # Banner
#        html.Div(
#            id="banner",
#            className="banner",
#            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
#        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # package importance plot
                html.Div(
                    id="package_imp_card",
                    children=[
                        html.H5("Differential Package/Option Analysis"),
                        html.Hr(),
                        html.Div(id='filtered_data', style={'display': 'none'}),
                        html.Div(id='filtered_take_rate_data', style={'display': 'none'}),
                        html.Div(id='feature_imp_data', style={'display': 'none'}),
                        
                        html.Div([html.Label('Take Rate (%) Slider', id='take-rate-slider-label'),
                                html.Br(),
                                html.Br(),
                                dcc.RangeSlider(
                                id='take-rate-slider',
                                min=0,
                                max=100,
                                step=0.5,
                                marks={i : '{}'.format(i) for i in range(0,100,10)},
                                value=[0, 100],
                                tooltip = {'always_visible': True}
                            ),
                                html.Br(),
                                html.Br(),
                                html.Div(id='sample_rate_text'),
                                dcc.Graph(id="package_imp_bar")], id = 'barchart-container'),

                        html.Br(),
                        html.Div(id="package_analysis_table", children=initialize_table()),
                    ],
                ),
               #  w/o package/feature analysis
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
               html.Br(),
                html.Div(
                    id="simulator_card",
                    style={"height": "300%"},
                    children=[
                        html.H5("Days to Turn Simulator"),
                        html.Hr(),
                        html.Div(id="dtt_simulator_table", children=initialize_simulator_table()),
                    ],
                ),
               html.Br(),
               html.Br(),
                html.Div(
                    id="comparator_card",
                    style={"height": "300%"},
                    children=[
                        html.H5("Comparative Analysis"),
                        html.Hr(),
                        dcc.Link('Please Click Here for the Comparator   ', style={"color": "red", "font-size":"20px"}, href='/dash-dtt-analytics/comparator', className="tab"),
#                        html.Div(id="dtt_simulator_table", children=initialize_simulator_table()),
                    ],
                ),
         
            ],
        ),
    
    ], className='page')


comparatorPage = html.Div(
    id="comparator-container",
    children=[

        html.Div(
            id="comparator-left-column",
            className="four columns",
            children = [
            html.Div(
                    id="comparator1",
                    children=[comparator1_description_card(), comparator1_generate_control_card()],
                ),

            html.Div(
                    id="comparator2",
                    children=[comparator2_description_card(), comparator2_generate_control_card()],
                ),],
            
            
        ),
        # Right column
        html.Div(
            id="comparator-right-column",
            className="eight columns",
            children=[
                # Patient Volume Heatmap
                html.Div(
                    id="comparator_overview_card",
                    children=[
                        html.H4("Overview"),
#                        html.Hr(),
                        html.Div(id='comparator_overview_table', children = initialize_comparator_overview_table()),
                        html.Br(),
                        html.Br(),
                    ],
                ),
                # Patient Wait time by Department
                html.Div(
                    id="comparator_standard_features_card",
                    children=[
                        html.H4("Standard Features - JDP Standardized Features"),
#                        html.Hr(),
                        html.Div(id='standard_features_data', style={'display': 'none'}),
                        html.Div(id="standard_features_len", children=[html.Br(),]),
                        html.Div(id="standard_features_dropdown", children=[html.Br(),]),
                        html.Br(),
                        html.Div(id="standard_features_table", children=[dash_table.DataTable(
                                id='standard_features_table_result',)]),
                        html.Br(),
                        html.Br(),
                         
                    ],
                ),
                
                html.Div(
                    id="comparator_non_standard_features_card",
                    children=[
                        html.H4("Optional Features Take Rate - JDP Standardized Features"),
#                        html.Hr(),
                        html.Div(id='non_standard_features_data', style={'display': 'none'}),
                        html.Div(id="non_standard_features_dropdown", children=[html.Br(),]),
                        html.Br(),
                        html.Div(id="non_standard_features_table", children=[dash_table.DataTable(
                                id='non_standard_features_table_result',)]),
                        html.Br(),
                        html.Br(),
                         
                    ],
                ),

               html.Div(
                    id="comparator_packages_options_card",
                    children=[
                        html.H4("Packages/Options - Similarity Matrix of Included Features"),
#                        html.Hr(),
                        html.Div(id='feature_imp1_data', style={'display': 'none'}),
                        html.Div(id='feature_imp2_data', style={'display': 'none'}),                       
                        html.Div(id = 'similarity_matrix_container', children = [dcc.Graph(id="comparator_similarity_matrix")], ),
                        html.Br(),
                        html.Div(id='pkg_features_table', children=[]),
                    ],
                ),

            ],
        ),
                    
    ],className='page')





noPage = html.Div([  # 404

    html.P(["404 Page not found"])

    ], className="no-page")



# Describe the layout, or the UI, of the app
app.layout = html.Div([

    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


# Update page
# # # # # # # # #
# detail in depth what the callback below is doing
# # # # # # # # #
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dash-dtt-analytics':
        return mainPage
    elif pathname == '/dash-dtt-analytics/comparator':
        return comparatorPage
#    elif pathname == '/dash-vanguard-report/portfolio-management':
#        return portfolioManagement
#    elif pathname == '/dash-vanguard-report/fees':
#        return feesMins
#    elif pathname == '/dash-vanguard-report/distributions':
#        return distributions
#    elif pathname == '/dash-vanguard-report/news-and-reviews':
#        return newsReviews
#    elif pathname == '/dash-vanguard-report/full-view':
#        return overview,pricePerformance,portfolioManagement,feesMins,distributions,newsReviews
    else:
        return noPage






@app.callback(
    Output("filtered_data", "children"),
    [Input('submit-button', 'n_clicks'), ],
    [State("make-select", "value"),
     State("model-select", "value"),
     State("modelyear-select", "value"),
     State("trim-select", "value"),
     State("color-select", "value"),
     State("region-select", "value"),
     State("time-select", "value"),
     ]
)
@cache.memoize(timeout=TIMEOUT)
def fitlered_data(n_clicks, make, model, modelyear, trim, color, region, sales_quarter):

    if n_clicks:
         all_sample_data = get_sample_data(make, model, modelyear, trim, color, region, sales_quarter)
         return all_sample_data.to_json(date_format='iso', orient='split')
    else:
         raise PreventUpdate


@app.callback(Output('feature_imp_data', 'children'), [Input('filtered_data', 'children'), Input('submit-button', 'n_clicks'),])
@cache.memoize(timeout=TIMEOUT)
def generate_feature_imp_data(jsonified_filtered_data, n_clicks):

    # more generally, this line would be
    # json.loads(jsonified_cleaned_data)
    
    if n_clicks:
        all_sample_data = pd.read_json(jsonified_filtered_data, orient='split')
        sample_data = all_sample_data.dropna(subset=['days_to_turn'])
        dtt_sr = sample_data.shape[0]
        
        if dtt_sr > 100:
            trim = sample_data['trim'].unique()[0]
            
            xx = sample_data.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
            'exterior_color', 'msrp', 'region', 'division', 'veh_gross_profit', 'incentives', 'cftp',
                    'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter', 'features'], axis = 1)
        
            pkg_list = xx.columns.tolist()
            pkg_list.remove('days_to_turn')
            
            rs_df = []
            rs_list = [245, 37875, 907, 65431, 719, 123456, 315, 74178, 468, 59815, 64, 9064, 312690, 80, 78412, 51, 159043, 12, 420978,
              55]
            for i in range(20):
                temp = xx.sample(frac=0.8, random_state=rs_list[i])
                temp_agg = temp.groupby(pkg_list).agg({'days_to_turn': 'mean'}).reset_index()
                rs_df.append(temp_agg)    
            
            df_rs = pd.concat(rs_df)
            df_rs_final = df_rs.drop_duplicates()
            
            X = df_rs_final.drop('days_to_turn', axis=1)
            y = df_rs_final['days_to_turn']
            
            ## Mann Whitney U significance test
            ddf_list = []
            for c in xx.columns[1:]:
                data1,data2 = (xx[xx[c]==0]['days_to_turn']),(xx[xx[c]==1]['days_to_turn'])
                ddf_list.append(mannwhitneyu_test(c,data1,data2))
        
            student_test_Mannwhitneyu = pd.DataFrame(ddf_list,columns=['Package','Mannwhitneyu_Stat',
                                                              'Mannwhitneyu_P_value'])
            
            non_sig = student_test_Mannwhitneyu[student_test_Mannwhitneyu['Mannwhitneyu_P_value'] >= 0.05]['Package'].tolist()
            
            ## Light GBM
            lr, md, ne = lgbm_param(trim)
            model = LGBMRegressor(random_state=0, n_estimators=ne, learning_rate=lr, max_depth=md)
            model.fit(X, y)
            
            dtt_change = []
            for feature in pkg_list:
                pdp_goals = pdp.pdp_isolate(model=model, dataset=X, model_features=pkg_list,
                                        feature=feature,num_grid_points=20)
                pdp_value = pdp.pdp_plot(pdp_goals, feature)
                
                if len(pdp_value[1]) == 1:
                    dtt_change.append(0)
                else:
                    dtt_change.append(pdp_value[1][1])
                    
            feature_imp = pd.DataFrame({'Feature': pkg_list, 'Value': dtt_change})
            feature_imp.sort_values(by='Value', inplace=True)
            feature_imp['color'] = feature_imp.apply(lambda row: color(row, non_sig), axis=1)
    #    
            package_take_rate = []
            package_take_rate_value = []
            package_count = []
            for package in feature_imp['Feature'].tolist():
                take_rate_value = 100.0 * all_sample_data[all_sample_data[package] == 1].shape[0] / all_sample_data.shape[0]
                take_rate_value = round(take_rate_value, 1)
                if take_rate_value != 'N/A':
                    take_rate_result = str(take_rate_value) + '%'
                else:
                    take_rate_result = take_rate_value
                package_take_rate.append(take_rate_result)
                package_take_rate_value.append(take_rate_value)
                package_count.append(all_sample_data[all_sample_data[package] == 1].shape[0])
        
            
            feature_imp['take_rate'] = package_take_rate
            feature_imp['take_rate_value'] = package_take_rate_value
            feature_imp['count'] = package_count
        
            return feature_imp.to_json(date_format='iso', orient='split')
        else:
            return None
    else:
        PreventUpdate

@app.callback(
    Output("sample_rate_text", "children"),
    [
        Input('filtered_data', 'children'),
        Input('submit-button', 'n_clicks'),
    ],
)

def update_sample_rate_text(jsonified_filtered_data, n_clicks):
      
    if n_clicks:
        all_sample_data = pd.read_json(jsonified_filtered_data, orient='split')
        sample_data = all_sample_data.dropna(subset=['days_to_turn'])
        
        sr = all_sample_data.shape[0]
        dtt_sr = sample_data.shape[0]
        
        return html.P('Sample Rate: ' + str(sr) + '  Days to Turn Sample Rate: ' + str(dtt_sr))
    else:
        PreventUpdate


@app.callback(
    Output("package_imp_bar", "figure"),
    [
        Input("feature_imp_data", "children"),
        Input('filtered_data', 'children'),
        Input('take-rate-slider', 'value'),
        Input("package_imp_bar", "clickData"),
        Input('submit-button', 'n_clicks'),
    ],
)

def update_barchart(jsonified_feature_imp_data, jsonified_filtered_data, take_rate_value, bar_click, n_clicks):
#    reset = False
    # Find which one has been triggered
      
    if n_clicks:
        ctx = dash.callback_context
    
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #        if prop_id == "reset-btn":
    #            reset = True
    
        all_sample_data = pd.read_json(jsonified_filtered_data, orient='split')
        sample_data = all_sample_data.dropna(subset=['days_to_turn'])
        
        if jsonified_feature_imp_data is not None:
            final_imp = pd.read_json(jsonified_feature_imp_data, orient='split')
            take_rate_min = take_rate_value[0]
            take_rate_max = take_rate_value[1]
            final_imp = final_imp[(final_imp['take_rate_value'] >= take_rate_min) & (final_imp['take_rate_value'] <= take_rate_max)]
            
            plot_data = pd.DataFrame({'plot_name': final_imp['Feature'], 'plot_value': final_imp['Value'], 'color': final_imp['color'], 'take_rate': final_imp['take_rate']})
            
            range_min = plot_data['plot_value'].min() - 10
            range_max = plot_data['plot_value'].max() + 10
            
    #        if sample_data['trim'].unique()[0] == '3.0 40i':
    #            plot_data['color'] = plot_data.apply(lambda row: color_bmw_40i(row), axis=1)
    #        else:
    #        plot_data['color'] = plot_data['plot_value'].apply(lambda x: color(x))
            
    #        package_list = final_imp['Feature'].unique().tolist()
    #        
    #        package_count = []
    #        for package in package_list:
    #            count = sample_data[sample_data[package] == 1].shape[0]
    #            package_count.append(count)
    #            
    #        plot_data['count'] = package_count
           
             
            ply_data = []
            for c in plot_data['color'].unique().tolist():
                temp = plot_data[plot_data['color'] == c]
                trace = dict(
                    x=temp['plot_value'],
                    y=temp['plot_name'],
                    type="bar",
                    name=color_legend[c],
                    text=temp['take_rate'],
                    textposition="outside",
                    cliponaxis=False,
                    marker=dict(
                    color=temp['color'].tolist(),
                    reversescale = True
                ),
                orientation='h',
                )
                
                ply_data.append(trace)
            
    #        trace2 = go.Scatter(
    #                    x=plot_data['take_rate'],
    #                    y=plot_data['plot_name'],
    #                    name='Take Rate',
    #                    xaxis='x2'
    #                )
    #        ply_data.append(trace2)
        
        
            layout = dict(
                title='Days to Turn: Differential Packages/Options',
                 width = 1000, height = 850,
                 legend=dict(orientation="h",
                             font=dict(size=16), y=-0.2),
                yaxis=dict(
                    showgrid=False,
                    showline=True,
                    linewidth=2, linecolor='black',
                    showticklabels=True,
                    automargin = True,mirror=True,
            #        tickfont = dict(color='green', family = "Arial", size = 13)
            #         domain=[0, 0.85],
                ),
                xaxis=dict(
                    showline=True,
                    showticklabels=True,
                    range=[range_min, range_max],
                    title='DTT Change',
                    linewidth=2, linecolor='black',
                    zeroline=True, zerolinewidth=4, zerolinecolor='LightPink',
                    mirror=True
            #         domain=[0, 0.85],
                ))
            
            figure=go.Figure(data=ply_data, layout=layout)
        
            
            return figure
        else:
            return go.Figure(data=[], layout= dict(annotations=[
                                            go.layout.Annotation(
                                                text='Days to Turn Sample Size is Too Small.',
                                                align='left',
                                                showarrow=False,
                                                xref='paper',
                                                yref='paper',
                                                x=0.5,
                                                y=0.5,
                                                font=dict(
                                                size=24,
                                                color = 'red'
                                                ),
                                            )
                                        ],))

    else:
        PreventUpdate


#app.clientside_callback(
#    ClientsideFunction(namespace="clientside", function_name="resize"),
#    Output("output-clientside", "children"),
#    [Input("package_analysis_table", "children")],
#)


@app.callback(
    Output("package_analysis_table", "children"),
    [
        Input("feature_imp_data", "children"),
        Input('filtered_data', 'children'),
        Input("package_imp_bar", "clickData"),
        Input("reset-btn", "n_clicks"),
        Input('submit-button', 'n_clicks'),
    ],
)

def update_table(jsonified_feature_imp_data, jsonified_filtered_data, bar_click, reset, n_clicks, *args):
 
              
    if n_clicks:
        reset = False
        # Find which one has been triggered
        ctx = dash.callback_context
    
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-btn":
                reset = True
    
    #    prop_id = ""
    #    prop_type = ""
    #    triggered_value = None
    #    if ctx.triggered:
    #        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #        prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
    #        triggered_value = ctx.triggered[0]["value"]
        
            
        all_sample_data = pd.read_json(jsonified_filtered_data, orient='split')
        sample_data = all_sample_data.dropna(subset=['days_to_turn'])
        
        if jsonified_feature_imp_data is not None:
            final_imp = pd.read_json(jsonified_feature_imp_data, orient='split')
            
            if reset == True:
                return initialize_table()
        
            # Highlight click data's patients in this table
            if bar_click is not None and prop_id != "reset-btn":
                target_package = bar_click["points"][0]["y"]
                
            
        #        differential_package = final_imp['Feature'].unique().tolist()
        #        
        #        df_base = sample_data.drop(['make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
        #     'exterior_color', 'region', 'division','veh_gross_profit', 'incentives', 'cftp',
        #                'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter'], axis=1)
        #        
        #        for i in differential_package:
        #            df_base = df_base[df_base[i] == 0]
        #        
        #        base_msrp = np.mean(df_base['msrp'])
        #        
        #        
        #        df_package = sample_data.drop(['make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
        #     'exterior_color', 'region', 'division','veh_gross_profit', 'incentives', 'cftp',
        #                'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter'], axis=1)
        #        
        #        for i in differential_package:
        #            if i == target_package:
        #                df_package = df_package[df_package[i] == 1]
        #            else:
        #                df_package = df_package[df_package[i] == 0]
        #    
        #        package_msrp = np.mean(df_package['msrp'])
        #    
        #    
        #        package_price = package_msrp - base_msrp
                
    
                if target_package in final_imp['Feature'].tolist():
                    
                    sample_size = final_imp[final_imp['Feature'] == target_package].iloc[0]['count']
                    take_rate = final_imp[final_imp['Feature'] == target_package].iloc[0]['take_rate']
                    dtt_change = round(final_imp[final_imp['Feature'] == target_package].iloc[0]['Value'],0)
                    
                    
    #                if target_package == 'Seat Trim-Upgraded Cloth/Velour':
    #                    dtt_w_package = np.mean(sample_data[sample_data[target_package] == 1]['days_to_turn'])
    #                    dtt_wo_package = np.mean(sample_data[sample_data[target_package] == 0]['days_to_turn'])
    #                    take_rate_value = 'N/A'
    #                else:
    #                    feature_count = final_imp.groupby('count')['Feature'].apply(list).reset_index()
    #                    feature_count['same_count'] = feature_count['Feature'].map(len)
    #                    feature_count.columns = ['count', 'same_feature_list', 'same_count']
    #                    same = feature_count[feature_count['same_count'] > 1]
    #                    same_package = pd.merge(final_imp, same, on = 'count', how='inner')
    #                    
    #                    groupby_list = final_imp['Feature'].unique().tolist()
    #                    groupby_list = [x for x in groupby_list if x != 'Seat Trim-Upgraded Cloth/Velour']
    #                    
    #                    if target_package in same_package['Feature'].unique():
    #                        remove_target_list = [x for x in groupby_list if x not in same_package[same_package['Feature'] == target_package]['same_feature_list'].iloc[0]]
    #                    else:
    #                        remove_target_list = [x for x in groupby_list if x != target_package]
    #                    
    #                    df_temp1 = sample_data.groupby(groupby_list)['days_to_turn'].mean().reset_index()
    #                    
    #                    df_temp2 = sample_data.groupby(remove_target_list).agg({'vin':'count', target_package: 'sum'}).reset_index()
    #                    df_temp2 = df_temp2[(df_temp2[target_package] < df_temp2['vin']) & (df_temp2[target_package] != 0)]
    #                    df_temp2.drop(['vin'], axis=1, inplace=True)
    #                    df_temp2.columns = remove_target_list + [target_package+'_count']
    #                    
    #                    df_temp = pd.merge(df_temp1, df_temp2, on = remove_target_list, how = 'inner')
    #    #                df_temp = df_temp[df_temp[target_package+'_count'] > 10]
    #                    
    #    #                w_package_df = df_temp[df_temp[target_package] == 1]
    #    #                dtt_w_package = (w_package_df[target_package+'_count']*w_package_df['days_to_turn']).sum() / w_package_df[target_package+'_count'].sum()
    #    #                wo_package_df = df_temp[df_temp[target_package] == 0]
    #    #                dtt_wo_package = (wo_package_df[target_package+'_count']*wo_package_df['days_to_turn']).sum() / wo_package_df[target_package+'_count'].sum()
    #                    
    #                    dtt_w_package = np.mean(df_temp[df_temp[target_package] == 1]['days_to_turn'])
    #                    dtt_wo_package = np.mean(df_temp[df_temp[target_package] == 0]['days_to_turn'])
    #                    
    #                    
    #    #                dtt_w_package = np.mean(sample_data[sample_data[target_package] == 1]['days_to_turn'])
    #    #                dtt_wo_package = np.mean(sample_data[sample_data[target_package] == 0]['days_to_turn'])
    #                    
    #                    take_rate_value = 100.0 * sample_data[sample_data[target_package] == 1].shape[0] / sample_data.shape[0]
    #                    take_rate_value = round(take_rate_value, 1)
    
                
        
                        
                    annotation = feature_mapping.groupby('package_option')['jdp_feature_name'].apply(lambda x: ', '.join(x)).reset_index()
                    
                    included_features = annotation[annotation['package_option'] == target_package].iloc[0]['jdp_feature_name']
                
                    
                
    #                # Put figures in table
    #                if take_rate_value != 'N/A':
    #                    take_rate_result = str(take_rate_value) + '%'
    #                else:
    #                    take_rate_result = take_rate_value
                        
                    table = generate_patient_table(
                        target_package, included_features, sample_size, take_rate, dtt_change)
                    return table

        else:
            PreventUpdate



@app.callback(
    Output("dtt_simulator_table", "children"),
    [
        Input("feature_imp_data", "children"),
#        Input('fitlered_data', 'children'),
#        Input("diff-package-select", "value"),
        Input("reset-btn", "n_clicks"),
        Input('submit-button', 'n_clicks'),
    ],
)

def update_simulator_table(jsonified_feature_imp_data, reset, n_clicks, *args):
   
    if n_clicks:
        reset = False
        # Find which one has been triggered
        ctx = dash.callback_context
    
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-btn":
                reset = True
    
    #    prop_id = ""
    #    prop_type = ""
    #    triggered_value = None
    #    if ctx.triggered:
    #        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #        prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
    #        triggered_value = ctx.triggered[0]["value"]
        if jsonified_feature_imp_data is not None:
            final_imp = pd.read_json(jsonified_feature_imp_data, orient='split')
        
            diff_packages = sorted(final_imp['Feature'].unique().tolist())
            
            table = generate_simulator_table(
                diff_packages
            )
            return table

    else:
        PreventUpdate



@app.callback(
    Output("sim_dtt", "children"),
    [
        Input("feature_imp_data", "children"),
        Input('filtered_data', 'children'),
        Input("diff-package-select", "value"),
        Input("reset-btn", "n_clicks"),
        Input('submit-button', 'n_clicks'),
    ],
)

def update_simulator_table_dtt(jsonified_feature_imp_data, jsonified_filtered_data, selected_packages, reset, n_clicks, *args):

   
    if n_clicks:
        reset = False
        # Find which one has been triggered
        ctx = dash.callback_context
    
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "reset-btn":
                reset = True
    
    #    prop_id = ""
    #    prop_type = ""
    #    triggered_value = None
    #    if ctx.triggered:
    #        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    #        prop_type = ctx.triggered[0]["prop_id"].split(".")[1]
    #        triggered_value = ctx.triggered[0]["value"]
        
            
        all_sample_data = pd.read_json(jsonified_filtered_data, orient='split')
        sample_data = all_sample_data.dropna(subset=['days_to_turn'])
        
        if jsonified_feature_imp_data is not None:
            final_imp = pd.read_json(jsonified_feature_imp_data, orient='split')
            
            df_sim = sample_data.drop(['make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
             'exterior_color', 'region', 'division','veh_gross_profit', 'incentives', 'cftp',
                        'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter'], axis=1)
            
        
            diff_packages = final_imp['Feature'].unique().tolist()
            
            for i in diff_packages:
                if i in selected_packages:
                    df_sim = df_sim[df_sim[i] == 1]
        
                else:
                    df_sim = df_sim[df_sim[i] == 0]
        
            
            dtt_sim = np.mean(df_sim['days_to_turn'])
            
            return round(dtt_sim,0)

    else:
        PreventUpdate

@app.callback(
    dash.dependencies.Output('model-select', 'options'),
    [dash.dependencies.Input('make-select', 'value')]
)
def update_model_dropdown(make):
    new_model_list = data[data['make'] == make]['model'].unique().tolist()
    return [{"label": i, "value": i} for i in new_model_list]

@app.callback(
    dash.dependencies.Output('modelyear-select', 'options'),
    [dash.dependencies.Input('make-select', 'value'),
     dash.dependencies.Input('model-select', 'value')]
)
def update_modelyear_dropdown(make, model):
    new_modelyear_list = data[(data['make'] == make) & (data['model'] == model)]['modelyear'].unique().tolist()
    return [{"label": i, "value": i} for i in new_modelyear_list]

@app.callback(
    dash.dependencies.Output('trim-select', 'options'),
    [dash.dependencies.Input('make-select', 'value'),
     dash.dependencies.Input('model-select', 'value'),
     dash.dependencies.Input('modelyear-select', 'value')]
)
def update_trim_dropdown(make, model, modelyear):
    new_trim_list = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear)]['trim'].unique().tolist()
    return [{"label": i, "value": i} for i in new_trim_list]

@app.callback(
    dash.dependencies.Output('color-select', 'options'),
    [dash.dependencies.Input('make-select', 'value'),
     dash.dependencies.Input('model-select', 'value'),
     dash.dependencies.Input('modelyear-select', 'value'),
     dash.dependencies.Input('trim-select', 'value'),]
)
def update_color_dropdown(make, model, modelyear, trim):
    new_color_list = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear) & (data['trim'] == trim)]['exterior_color'].unique().tolist()
    new_color_list = ['All Colors'] + new_color_list
    return [{"label": i, "value": i} for i in new_color_list]

@app.callback(
    dash.dependencies.Output('region-select', 'options'),
    [dash.dependencies.Input('make-select', 'value'),
     dash.dependencies.Input('model-select', 'value'),
     dash.dependencies.Input('modelyear-select', 'value'),
     dash.dependencies.Input('trim-select', 'value'),
     dash.dependencies.Input('color-select', 'value'),]
)
def update_region_dropdown(make, model, modelyear, trim, color):
    temp_data = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear) & (data['trim'] == trim)]
    
    if color != 'All Colors':
        temp_data = temp_data[temp_data['exterior_color'] == color]
        
    new_region_list = ['National'] + temp_data['region'].unique().tolist()
    return [{"label": i, "value": i} for i in new_region_list]

@app.callback(
    dash.dependencies.Output('time-select', 'options'),
    [dash.dependencies.Input('make-select', 'value'),
     dash.dependencies.Input('model-select', 'value'),
     dash.dependencies.Input('modelyear-select', 'value'),
     dash.dependencies.Input('trim-select', 'value'),
     dash.dependencies.Input('color-select', 'value'),
     dash.dependencies.Input('region-select', 'value'),]
)
def update_time_dropdown(make, model, modelyear, trim, color, region):
    temp_data = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear) & (data['trim'] == trim)]
    
    if color != 'All Colors':
        temp_data = temp_data[temp_data['exterior_color'] == color]
    
    if region != 'National':
        temp_data = temp_data[temp_data['region'] == region]
    
    new_time_list = sorted(temp_data['sales_quarter'].unique().tolist())
    
    return [{"label": i, "value": i} for i in new_time_list]


## comparator page callbacks
@app.callback(
    dash.dependencies.Output('comparator1-model-select', 'options'),
    [dash.dependencies.Input('comparator1-make-select', 'value')]
)
def update_comparator1_model_dropdown(make):
    new_model_list = data[data['make'] == make]['model'].unique().tolist()
    return [{"label": i, "value": i} for i in new_model_list]

@app.callback(
    dash.dependencies.Output('comparator1-modelyear-select', 'options'),
    [dash.dependencies.Input('comparator1-make-select', 'value'),
     dash.dependencies.Input('comparator1-model-select', 'value')]
)
def update_comparator1_modelyear_dropdown(make, model):
    new_modelyear_list = data[(data['make'] == make) & (data['model'] == model)]['modelyear'].unique().tolist()
    return [{"label": i, "value": i} for i in new_modelyear_list]

@app.callback(
    dash.dependencies.Output('comparator1-trim-select', 'options'),
    [dash.dependencies.Input('comparator1-make-select', 'value'),
     dash.dependencies.Input('comparator1-model-select', 'value'),
     dash.dependencies.Input('comparator1-modelyear-select', 'value')]
)
def update_comparator1_trim_dropdown(make, model, modelyear):
    new_trim_list = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear)]['trim'].unique().tolist()
    return [{"label": i, "value": i} for i in new_trim_list]

#@app.callback(
#    Output("comparator1", "children"),
#    [Input('comparator-submit-button', 'n_clicks'), ],
#    [State("comparator1-make-select", "value"),
#     State("comparator1-model-select", "value"),
#     State("comparator1-modelyear-select", "value"),
#     State("comparator1-trim-select", "value"),
#     ]
#)
#def update_comparator1(n_clicks, make, model, modelyear, trim):
#
#    if n_clicks:
##         return html.Div(
##            id="comparator1",
##            children=[comparator1_selected_card(make, model, modelyear, trim)]
##        ),
#        return comparator1_selected_card(make, model, modelyear, trim)
#    else:
#         raise PreventUpdate



@app.callback(
    dash.dependencies.Output('comparator2-model-select', 'options'),
    [dash.dependencies.Input('comparator2-make-select', 'value')]
)
def update_comparator2_model_dropdown(make):
    new_model_list = data[data['make'] == make]['model'].unique().tolist()
    return [{"label": i, "value": i} for i in new_model_list]

@app.callback(
    dash.dependencies.Output('comparator2-modelyear-select', 'options'),
    [dash.dependencies.Input('comparator2-make-select', 'value'),
     dash.dependencies.Input('comparator2-model-select', 'value')]
)
def update_comparator2_modelyear_dropdown(make, model):
    new_modelyear_list = data[(data['make'] == make) & (data['model'] == model)]['modelyear'].unique().tolist()
    return [{"label": i, "value": i} for i in new_modelyear_list]

@app.callback(
    dash.dependencies.Output('comparator2-trim-select', 'options'),
    [dash.dependencies.Input('comparator2-make-select', 'value'),
     dash.dependencies.Input('comparator2-model-select', 'value'),
     dash.dependencies.Input('comparator2-modelyear-select', 'value')]
)
def update_comparator2_trim_dropdown(make, model, modelyear):
    new_trim_list = data[(data['make'] == make) & (data['model'] == model) & (data['modelyear'] == modelyear)]['trim'].unique().tolist()
    return [{"label": i, "value": i} for i in new_trim_list]



# =============================================================================
# @app.callback(
#     Output("comparator2", "children"),
#     [Input('comparator-submit-button', 'n_clicks'), ],
#     [State("comparator2-make-select", "value"),
#      State("comparator2-model-select", "value"),
#      State("comparator2-modelyear-select", "value"),
#      State("comparator2-trim-select", "value"),
#      ]
# )
# def update_comparator2(n_clicks, make2, model2, modelyear2, trim2):
# 
#     if n_clicks:
# #         return html.Div(
# #            id="comparator2",
# #            children=[comparator_description_card(make2, model2, modelyear2)]
# #        ),
#         return comparator2_selected_card(make2, model2, modelyear2, trim2)
#     else:
#          raise PreventUpdate
# =============================================================================



@app.callback(
    Output("comparator_overview_table", "children"),
    [Input('comparator-submit-button', 'n_clicks'), 
#     Input('comparator1-trim-select', 'value'),
#     Input('comparator2-trim-select', 'value')
],
    [State("comparator1-make-select", "value"),
     State("comparator1-model-select", "value"),
     State("comparator1-modelyear-select", "value"),
     State("comparator1-trim-select", "value"),
     State("comparator2-make-select", "value"),
     State("comparator2-model-select", "value"),
     State("comparator2-modelyear-select", "value"),
     State("comparator2-trim-select", "value"),
     ]
)
@cache.memoize(timeout=TIMEOUT)
def update_comparator_overview_table(n_clicks, make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):

    if n_clicks:
        car1 = str(modelyear1) + ' ' + make1 + ' ' + model1 + ': ' + trim1
        car2 = str(modelyear2) + ' ' + make2 + ' ' + model2 + ': ' + trim2
        header = [
            generate_comparator_overview_table_row(
                "header",
                {"height": "50px"},
                {"id": "header_category", "children": html.H6("")},
                {"id": "header_model1", "children": html.H6(car1)},
                {"id": "header_model2", "children": html.H6(car2)},
            )
        ]
        
        overview_parameters = comparator_overview(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2)
        
        for key, value in overview_parameters.items():    
            row = [
                generate_comparator_overview_table_row(
                    "comparator_overviw_section",
                    {},
                    {"id": "category", "children": html.H6(key)},
                    {"id": "model1", "children": html.H6(value[0])},
                    {"id": "model2", "children": html.H6(value[1])},
    
                )
            ]
            
    
            header.extend(row)
        
        return header
    else:
         raise PreventUpdate


@app.callback(
    [Output("standard_features_dropdown", "children"), Output("standard_features_data", "children"), Output("standard_features_len", "children")],
    [Input('comparator-submit-button', 'n_clicks'), 
#     Input('comparator1-trim-select', 'value'),
#     Input('comparator2-trim-select', 'value')
],
    [State("comparator1-make-select", "value"),
     State("comparator1-model-select", "value"),
     State("comparator1-modelyear-select", "value"),
     State("comparator1-trim-select", "value"),
     State("comparator2-make-select", "value"),
     State("comparator2-model-select", "value"),
     State("comparator2-modelyear-select", "value"),
     State("comparator2-trim-select", "value"),
     ]
)
@cache.memoize(timeout=TIMEOUT)
def update_standard_features_dropdown(n_clicks, make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):

    if n_clicks:
        car1 = str(modelyear1) + ' ' + make1 + ' ' + model1 + ': ' + trim1
        car2 = str(modelyear2) + ' ' + make2 + ' ' + model2 + ': ' + trim2
        len1, len2, sf_df = comparator_standard_features(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2)
        category_list = sf_df['group'].unique().tolist()
        
        sf_df.columns = ['Category', 'Feature', car1, car2]

        row = [
            generate_comparator_overview_table_row(
                "comparator_standard_features_len_section",
                {},
                {"id": "category", "children": html.H6('Number of Standard Features')},
                {"id": "model1", "children": html.H6(len1)},
                {"id": "model2", "children": html.H6(len2)},

            )
        ]
        
        sf_dropdown = dcc.Dropdown(
                id="comparator-category-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in category_list],
                value=category_list[0],
            )
        
        
        return sf_dropdown, sf_df.to_json(date_format='iso', orient='split'), row
    else:
        raise PreventUpdate





@app.callback(
    Output("standard_features_table", "children"),
    [ Input('standard_features_data', 'children'),
     Input('comparator-category-select', 'value')
],

)
def update_standard_features_table(jsonified_standard_features_data, category):
    
    if jsonified_standard_features_data is not None:
    
        sf_df = pd.read_json(jsonified_standard_features_data, orient='split')
    
        sf_df_result = sf_df[sf_df['Category'] == category]
        sf_df_result.reset_index(drop=True, inplace=True)
        
        column_name = sf_df_result.columns.tolist()
        
        standard_features_table = dash_table.DataTable(
                                    id='standard_features_table_result',
                                    style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto'
                                },
                                    style_cell={'textAlign': 'center'},
                                    columns=[{"name": i, "id": i} for i in sf_df_result.columns],
                                    data=sf_df_result.to_dict('records'),
                                    
    #                                style_data_conditional=[
    #                                {
    #                                    'if': {'row_index': 'odd'},
    #                                    'backgroundColor': 'rgb(248, 248, 248)'
    #                                },
    #                                {
    #                                'if': {
    #                                    'filter': '{} eq "no"'.format('{'+str(column_name[2])+'}'),
    #                                },
    #                                'backgroundColor': 'pink'
    #                            },
    #                            {
    #                                'if': {
    #                                    'filter': '{} eq "no"'.format(str(column_name[3])),
    #                                },
    #                                'backgroundColor': 'lightblue'
    #                            },
    #                            ],
                       style_data_conditional = [{'if': {"row_index": x},'backgroundColor': 'pink'} for x in sf_df_result[(sf_df_result[column_name[2]]!='yes') | (sf_df_result[column_name[3]]!='yes')].index.tolist()],
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold',
                                }
                                    
                                )
        
    
        
        return standard_features_table

### Non standard features take rate callbacks
@app.callback(
    [Output("non_standard_features_dropdown", "children"), Output("non_standard_features_data", "children")],
    [Input('comparator-submit-button', 'n_clicks'), 
#     Input('comparator1-trim-select', 'value'),
#     Input('comparator2-trim-select', 'value')
],
    [State("comparator1-make-select", "value"),
     State("comparator1-model-select", "value"),
     State("comparator1-modelyear-select", "value"),
     State("comparator1-trim-select", "value"),
     State("comparator2-make-select", "value"),
     State("comparator2-model-select", "value"),
     State("comparator2-modelyear-select", "value"),
     State("comparator2-trim-select", "value"),
     ]
)
@cache.memoize(timeout=TIMEOUT)
def update_non_standard_features_dropdown(n_clicks, make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):

    if n_clicks:
        car1 = str(modelyear1) + ' ' + make1 + ' ' + model1 + ': ' + trim1
        car2 = str(modelyear2) + ' ' + make2 + ' ' + model2 + ': ' + trim2
        nsf_df = comparator_non_standard_features_take_rate(make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2)
        nsf_category_list = nsf_df['group'].unique().tolist()
        
        nsf_df.columns = ['Category', 'Feature', car1, car2]
        
        nsf_dropdown = dcc.Dropdown(
                id="comparator-nsf-category-select",
                style={'height': '30px', 'width': '400px'},
                options=[{"label": i, "value": i} for i in nsf_category_list],
                value=nsf_category_list[0],
            )
        
        
        return nsf_dropdown, nsf_df.to_json(date_format='iso', orient='split')
    else:
        raise PreventUpdate


@app.callback(
    Output("non_standard_features_table", "children"),
    [ Input('non_standard_features_data', 'children'),
     Input('comparator-nsf-category-select', 'value')
],

)
def update_non_standard_features_table(jsonified_non_standard_features_data, nsf_category):
    
    if jsonified_non_standard_features_data is not None:
    
        nsf_df = pd.read_json(jsonified_non_standard_features_data, orient='split')
    
        nsf_df_result = nsf_df[nsf_df['Category'] == nsf_category]
        nsf_df_result.reset_index(drop=True, inplace=True)
        
        non_standard_features_table = dash_table.DataTable(
                                    id='non_standard_features_table_result',
                                    style_data={
                                    'whiteSpace': 'normal',
                                    'height': 'auto'
                                },
                                    style_cell={'textAlign': 'center'},
                                    columns=[{"name": i, "id": i} for i in nsf_df_result.columns],
                                    data=nsf_df_result.to_dict('records'),
                                    
    #                                style_data_conditional=[
    #                                {
    #                                    'if': {'row_index': 'odd'},
    #                                    'backgroundColor': 'rgb(248, 248, 248)'
    #                                },
    #                                {
    #                                'if': {
    #                                    'filter': '{} eq "no"'.format('{'+str(column_name[2])+'}'),
    #                                },
    #                                'backgroundColor': 'pink'
    #                            },
    #                            {
    #                                'if': {
    #                                    'filter': '{} eq "no"'.format(str(column_name[3])),
    #                                },
    #                                'backgroundColor': 'lightblue'
    #                            },
    #                            ],
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold',
                                }
                                    
                                )
        
    
        
        return non_standard_features_table




## comparative analysis callbacks
@app.callback(
    [Output("comparator_similarity_matrix", "figure"), Output("feature_imp1_data", "children"), Output("feature_imp2_data", "children")],
    [Input('comparator-submit-button', 'n_clicks'), 
     Input("comparator_similarity_matrix", "clickData"),
#     Input('comparator1-trim-select', 'value'),
#     Input('comparator2-trim-select', 'value')
],
    [State("comparator1-make-select", "value"),
     State("comparator1-model-select", "value"),
     State("comparator1-modelyear-select", "value"),
     State("comparator1-trim-select", "value"),
     State("comparator2-make-select", "value"),
     State("comparator2-model-select", "value"),
     State("comparator2-modelyear-select", "value"),
     State("comparator2-trim-select", "value"),
     ]
)
@cache.memoize(timeout=TIMEOUT)
def update_comparator_similarity_matrix(n_clicks, bar_click, make1, model1, modelyear1, trim1, make2, model2, modelyear2, trim2):

    if n_clicks:
        car1 = str(modelyear1) + ' ' + make1 + ' ' + model1 + ': ' + trim1
        car2 = str(modelyear2) + ' ' + make2 + ' ' + model2 + ': ' + trim2
        
        ## generate feature importance data

        comparator1_df = data[(data['make'] == make1) & (data['model'] == model1) & (data['modelyear'] == modelyear1) & (data['trim'] == trim1)]
        comparator2_df = data[(data['make'] == make2) & (data['model'] == model2) & (data['modelyear'] == modelyear2) & (data['trim'] == trim2)]
        
        # packages
        list_packs_comparator1 = []
        unique_packs_comparator1 = []
        def sortPacksList_comparator1(i):
            if not i: return None
            else:
                i.sort()
                if i not in list_packs_comparator1:
                    list_packs_comparator1.append(i)
                    for j in i:
                        if j not in unique_packs_comparator1:
                            unique_packs_comparator1.append(j)
            return i
        comparator1_df['packages'].apply(sortPacksList_comparator1)
        
        list_packs_comparator2 = []
        unique_packs_comparator2 = []
        def sortPacksList_comparator1(i):
            if not i: return None
            else:
                i.sort()
                if i not in list_packs_comparator2:
                    list_packs_comparator2.append(i)
                    for j in i:
                        if j not in unique_packs_comparator2:
                            unique_packs_comparator2.append(j)
            return i
        comparator2_df['packages'].apply(sortPacksList_comparator1)
    
        sample_data1 = ohe_features(comparator1_df, 'packages')
        sample_data1 = ohe_features(sample_data1, 'options')
        sample_data1.dropna(subset=['days_to_turn'], inplace=True)
        dtt_sr1 = sample_data1.shape[0]
        
        sample_data2 = ohe_features(comparator2_df, 'packages')
        sample_data2 = ohe_features(sample_data2, 'options')
        sample_data2.dropna(subset=['days_to_turn'], inplace=True)
        dtt_sr2 = sample_data2.shape[0]
        
        if dtt_sr1 > 100 and dtt_sr2 > 100:
        
            rs_list = [245, 37875, 907, 65431, 719, 123456, 315, 74178, 468, 59815, 64, 9064, 312690, 80, 78412, 51, 159043, 12, 420978,
              55]
            
            ### generate feature importance for comparator1
            xx1 = sample_data1.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
            'exterior_color', 'msrp', 'region', 'division', 'veh_gross_profit', 'incentives', 'cftp',
                    'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter', 'features'], axis = 1)
        
            pkg_list1 = xx1.columns.tolist()
            pkg_list1.remove('days_to_turn')
            
            rs_df1 = []
            for i in range(20):
                temp = xx1.sample(frac=0.8, random_state=rs_list[i])
                temp_agg = temp.groupby(pkg_list1).agg({'days_to_turn': 'mean'}).reset_index()
                rs_df1.append(temp_agg)    
            
            df_rs1 = pd.concat(rs_df1)
            df_rs_final1 = df_rs1.drop_duplicates()
            
            X1 = df_rs_final1.drop('days_to_turn', axis=1)
            y1 = df_rs_final1['days_to_turn']
            
            ## Mann Whitney U significance test
            ddf_list1 = []
            for c in xx1.columns[1:]:
                data1,data2 = (xx1[xx1[c]==0]['days_to_turn']),(xx1[xx1[c]==1]['days_to_turn'])
                ddf_list1.append(mannwhitneyu_test(c,data1,data2))
        
            student_test_Mannwhitneyu1 = pd.DataFrame(ddf_list1,columns=['Package','Mannwhitneyu_Stat',
                                                              'Mannwhitneyu_P_value'])
            
            non_sig1 = student_test_Mannwhitneyu1[student_test_Mannwhitneyu1['Mannwhitneyu_P_value'] >= 0.05]['Package'].tolist()
            
            ## Light GBM
            lr1, md1, ne1 = lgbm_param(trim1)
            model1 = LGBMRegressor(random_state=0, n_estimators=ne1, learning_rate=lr1, max_depth=md1)
            model1.fit(X1, y1)
            
            dtt_change1 = []
            for feature in pkg_list1:
                pdp_goals1 = pdp.pdp_isolate(model=model1, dataset=X1, model_features=pkg_list1,
                                        feature=feature,num_grid_points=20)
                pdp_value1 = pdp.pdp_plot(pdp_goals1, feature)
                
                if len(pdp_value1[1]) == 1:
                    dtt_change1.append(0)
                else:
                    dtt_change1.append(pdp_value1[1][1])
                    
            feature_imp1 = pd.DataFrame({'package_option': pkg_list1, 'value': dtt_change1})
            feature_imp1.sort_values(by='value', inplace=True)
            feature_imp1['color'] = feature_imp1.apply(lambda row: color_comparator(row, non_sig1), axis=1)
            
            ### generate feature importance for comparator2
            xx2 = sample_data2.drop(['vin', 'make', 'model', 'modelyear', 'trim', 'drivetrain', 'veh_fuel_type', 'liters', 'cylinders', 'doors',
            'exterior_color', 'msrp', 'region', 'division', 'veh_gross_profit', 'incentives', 'cftp',
                    'transaction_year', 'transaction_month', 'sales_year_month', 'sales_quarter', 'features'], axis = 1)
        
            pkg_list2 = xx2.columns.tolist()
            pkg_list2.remove('days_to_turn')
            
            rs_df2 = []
            for i in range(20):
                temp = xx2.sample(frac=0.8, random_state=rs_list[i])
                temp_agg = temp.groupby(pkg_list2).agg({'days_to_turn': 'mean'}).reset_index()
                rs_df2.append(temp_agg)    
            
            df_rs2 = pd.concat(rs_df2)
            df_rs_final2 = df_rs2.drop_duplicates()
            
            X2 = df_rs_final2.drop('days_to_turn', axis=1)
            y2 = df_rs_final2['days_to_turn']
            
            ## Mann Whitney U significance test
            ddf_list2 = []
            for c in xx2.columns[1:]:
                data1,data2 = (xx2[xx2[c]==0]['days_to_turn']),(xx2[xx2[c]==1]['days_to_turn'])
                ddf_list2.append(mannwhitneyu_test(c,data1,data2))
        
            student_test_Mannwhitneyu2 = pd.DataFrame(ddf_list2,columns=['Package','Mannwhitneyu_Stat',
                                                              'Mannwhitneyu_P_value'])
            
            non_sig2 = student_test_Mannwhitneyu2[student_test_Mannwhitneyu2['Mannwhitneyu_P_value'] >= 0.05]['Package'].tolist()
            
            ## Light GBM
            lr2, md2, ne2 = lgbm_param(trim2)
            model2 = LGBMRegressor(random_state=0, n_estimators=ne2, learning_rate=lr2, max_depth=md2)
            model2.fit(X2, y2)
            
            dtt_change2 = []
            for feature in pkg_list2:
                pdp_goals2 = pdp.pdp_isolate(model=model2, dataset=X2, model_features=pkg_list2,
                                        feature=feature,num_grid_points=20)
                pdp_value2 = pdp.pdp_plot(pdp_goals2, feature)
                
                if len(pdp_value2[1]) == 1:
                    dtt_change2.append(0)
                else:
                    dtt_change2.append(pdp_value2[1][1])
                    
            feature_imp2 = pd.DataFrame({'package_option': pkg_list2, 'value': dtt_change2})
            feature_imp2.sort_values(by='value', inplace=True)
            feature_imp2['color'] = feature_imp2.apply(lambda row: color_comparator(row, non_sig2), axis=1)
            
            
            feature_imp1['effect'] = feature_imp1['color'].apply(effect)
            feature_imp2['effect'] = feature_imp2['color'].apply(effect)
            
            feature_imp1 = pd.merge(feature_imp1, feature_mapping, on='package_option', how='left')
            feature_imp1['category'] = feature_imp1['package_option'].apply(lambda x: 'P' if x in unique_packs_comparator1 else 'O')
            
            feature_imp2 = pd.merge(feature_imp2, feature_mapping, on='package_option', how='left')
            feature_imp2['category'] = feature_imp2['package_option'].apply(lambda x: 'P' if x in unique_packs_comparator2 else 'O')
            
            feature_imp1['car1'] = car1
            feature_imp2['car2'] = car2
            
            ### generate similarity matrixx
            feature_imp1_temp = feature_imp1.dropna(subset=['jdp_feature_code'])
            feature_imp1_temp['jdp_feature_code'] = feature_imp1_temp['jdp_feature_code'].apply(int)
            feature_imp1_agg = feature_imp1_temp.groupby('package_option').agg({'jdp_feature_code': list, 'jdp_feature_name': list,
                                                                          'effect': 'first', 'category': 'first', 'value': 'first'}).reset_index()
            feature_imp1_agg.sort_values(by='value', inplace=True)
            
            feature_imp2_temp = feature_imp2.dropna(subset=['jdp_feature_code'])
            feature_imp2_temp['jdp_feature_code'] = feature_imp2_temp['jdp_feature_code'].apply(int)
            feature_imp2_agg = feature_imp2_temp.groupby('package_option').agg({'jdp_feature_code': list, 'jdp_feature_name': list,
                                                                          'effect': 'first', 'category': 'first', 'value': 'first'}).reset_index()
            feature_imp2_agg.sort_values(by='value', inplace=True)
            
            comparator1_positive = feature_imp1_agg[feature_imp1_agg['effect'] == 'Positively Impact DTT']['package_option'].tolist()
            comparator1_negative = feature_imp1_agg[feature_imp1_agg['effect'] == 'Negatively Impact DTT']['package_option'].tolist()
            comparator1_notsig = feature_imp1_agg[feature_imp1_agg['effect'] == 'Not Significant']['package_option'].tolist()
            feature_imp1_agg_new = pd.concat([feature_imp1_agg[feature_imp1_agg['effect'] == 'Positively Impact DTT'], feature_imp1_agg[feature_imp1_agg['effect'] == 'Negatively Impact DTT'],
                                              feature_imp1_agg[feature_imp1_agg['effect'] == 'Not Significant']])
            
            comparator2_positive = feature_imp2_agg[feature_imp2_agg['effect'] == 'Positively Impact DTT']['package_option'].tolist()
            comparator2_negative = feature_imp2_agg[feature_imp2_agg['effect'] == 'Negatively Impact DTT']['package_option'].tolist()
            comparator2_notsig = feature_imp2_agg[feature_imp2_agg['effect'] == 'Not Significant']['package_option'].tolist()
            feature_imp2_agg_new = pd.concat([feature_imp2_agg[feature_imp2_agg['effect'] == 'Positively Impact DTT'], feature_imp2_agg[feature_imp2_agg['effect'] == 'Negatively Impact DTT'],
                                              feature_imp2_agg[feature_imp2_agg['effect'] == 'Not Significant']])
            
            comparator1_pkg_option_list = comparator1_positive + comparator1_negative + comparator1_notsig
            comparator2_pkg_option_list = comparator2_positive + comparator2_negative + comparator2_notsig
            
            m = len(comparator1_pkg_option_list)
            n= len(comparator2_pkg_option_list)
            M_similarity = np.zeros((m, n))
        
            #Iterate through DataFrame columns to measure similarity
            for i in range(m):
                u = feature_imp1_agg_new.iloc[i]['jdp_feature_code']
                for j in range(n):
                    v = feature_imp2_agg_new.iloc[j]['jdp_feature_code']
                    M_similarity[i,j] = common_elements(u, v)
            DF_similarity = pd.DataFrame(M_similarity,columns=comparator2_pkg_option_list,index=comparator1_pkg_option_list)
            
            hovertext = list()
            for yi, yy in enumerate(comparator1_pkg_option_list):
                hovertext.append(list())
                for xi, xx in enumerate(comparator2_pkg_option_list):
                    z = int(M_similarity[yi][xi])
                    hovertext[-1].append(car2 + ": {}<br>".format(xx) +
                car1 + ": {}<br>".format(yy) +
                "Number of Common Features: {}".format(z))
            
            hm_shapes = []
            nx = len(comparator2_pkg_option_list)
            ny = len(comparator1_pkg_option_list)
            
            for i in np.arange(0.5, nx, 1):
                hm_shapes.append(dict(type='line', x0=i, y0=0, x1=i, y1=ny,
                                                        line=dict(color='White', width=1)))
            for j in np.arange(0.5, ny, 1):
                hm_shapes.append(dict(type='line', x0=0, y0=j, x1=nx, y1=j,
                                                        line=dict(color='White', width=1)))
       
                
            plot_data=[go.Heatmap(z=DF_similarity.values,
                        x=comparator2_pkg_option_list,
                        y=comparator1_pkg_option_list, colorscale='ice',
                        hoverinfo='text',
                        text=hovertext
    ), 
               go.Heatmap(yaxis='y2'), go.Heatmap(yaxis='y3'), go.Heatmap(xaxis='x2'), go.Heatmap(xaxis='x3')]
        
            layout = dict(width = 1500, height = 930,
    #                                        title= dict(text='Similarity Matrix of Included Features',
    #                                                    font=dict(size=20),
    #                                                    x=0.2,
    #                                                    y=0.98,
    #                                                   xanchor='center',
    #                                                   yanchor='top'),
                                            annotations=[
                                                go.layout.Annotation(
                                                    text='Green: Positively Impact DTT<br>Red: Negatively Impact DTT<br>Grey: Not Significant',
                                                    align='left',
                                                    showarrow=False,
                                                    xref='paper',
                                                    yref='paper',
                                                    x=-0.5,
                                                    y=1.3,
                                                    bordercolor='black',
                                                    borderwidth=1,
                                                    font=dict(
                                                    size=16,
                                                    ),
                                                )
                                            ],
                                            shapes=hm_shapes,
                                                        
                                            yaxis=dict(
                                            range = [0, len(comparator1_pkg_option_list)-1],
    #                                        title=dict(text=car1, font=dict(size=20)),
                                            tickfont=dict(color="green"),
                                            tickmode='array',
                                            tickvals=[i for i in range(0, len(comparator1_positive))],
                                            ticktext=comparator1_positive,
                                            automargin = True,mirror=True,
                                        ),
                                        yaxis2=dict(
                                            range = [0, len(comparator1_pkg_option_list)-1],
                                            title=dict(text=car1, font=dict(size=20)),
                                            tickfont=dict(color="red"),
                                            tickmode='array',
                                            tickvals=[i for i in range(len(comparator1_positive), len(comparator1_positive)+len(comparator1_negative))],
                                            ticktext=comparator1_negative,
                                            automargin = True,mirror=True,
                                            overlaying="y",
                                            side='left'
                                        ),
                                        yaxis3=dict(
                                            range = [0, len(comparator1_pkg_option_list)-1],
                                            tickfont=dict(color="grey"),
                                            tickmode='array',
                                            tickvals=[i for i in range(len(comparator1_positive)+len(comparator1_negative), len(comparator1_pkg_option_list))],
                                            ticktext=comparator1_notsig,
                                            automargin = True,mirror=True,
                                            overlaying="y",
                                            side='left'
                                        ),
                                        xaxis=dict(
                                            range = [0, len(comparator2_pkg_option_list)-1],
    #                                        title=dict(text=car2, font=dict(size=20)),
                                            side = 'top',
                                            tickfont=dict(color="green"),
                                            tickangle=-90,
                                            tickmode='array',
                                            tickvals=[i for i in range(0, len(comparator2_positive))],
                                            ticktext=[splitTextonWords(x,3) for x in comparator2_positive],
                                            automargin = True,mirror=True,
                                        ),
                                        xaxis2=dict(
                                            range = [0, len(comparator2_pkg_option_list)-1],
                                            title=dict(text=car2, font=dict(size=20)),
                                            side = 'top',
                                            tickfont=dict(color="red"),
                                            tickangle=-90,
                                            tickmode='array',
                                            tickvals=[i for i in range(len(comparator2_positive), len(comparator2_positive)+len(comparator2_negative))],
                                            ticktext=[splitTextonWords(x,3) for x in comparator2_negative],
                                            automargin = True,mirror=True,
                                            overlaying="x",
                                        ),
                                        xaxis3=dict(
                                            range = [0, len(comparator2_pkg_option_list)-1],
                                            side = 'top',
                                            tickfont=dict(color="grey"),
                                            tickangle=-90,
                                            tickmode='array',
                                            tickvals=[i for i in range(len(comparator2_positive)+len(comparator2_negative), len(comparator2_pkg_option_list))],
                                            ticktext=[splitTextonWords(x,3) for x in comparator2_notsig],
                                            automargin = True,mirror=True,
                                            overlaying="x",
                                        ))
                                        
            figure=go.Figure(data=plot_data, layout=layout)
            
            
            return figure, feature_imp1.to_json(date_format='iso', orient='split'), feature_imp2.to_json(date_format='iso', orient='split')
        else:
            figure = go.Figure(data=[], layout= dict(annotations=[
                                            go.layout.Annotation(
                                                text='Days to Turn Sample Size is Too Small.',
                                                align='left',
                                                showarrow=False,
                                                xref='paper',
                                                yref='paper',
                                                x=0.5,
                                                y=0.5,
                                                font=dict(
                                                size=24,
                                                color = 'red'
                                                ),
                                            )
                                        ],))
            return figure, None, None
    else:
        raise PreventUpdate


@app.callback(
    Output("pkg_features_table", "children"),
    [ Input('feature_imp1_data', 'children'),
      Input('feature_imp2_data', 'children'),
     Input("comparator_similarity_matrix", "clickData"),
     Input('comparator-submit-button', 'n_clicks')
],

)

def update_pkg_features_table(jsonified_feature_imp1_data, jsonified_feature_imp2_data, bar_click, n_clicks):
    
    if n_clicks:

        if bar_click is not None:
            
            comparator1_pkg = bar_click["points"][0]["y"]
            comparator2_pkg = bar_click["points"][0]["x"]
            
            if jsonified_feature_imp1_data is not None and jsonified_feature_imp2_data is not None:
        
                feature_imp1 = pd.read_json(jsonified_feature_imp1_data, orient='split')
                feature_imp2 = pd.read_json(jsonified_feature_imp2_data, orient='split')
                
                if comparator1_pkg in feature_imp1['package_option'].unique() and comparator2_pkg in feature_imp2['package_option'].unique():
                
                    feature_imp1_selected = feature_imp1[feature_imp1['package_option'] == comparator1_pkg]
                    feature_imp2_selected = feature_imp2[feature_imp2['package_option'] == comparator2_pkg]
                    
                    car1 = feature_imp1['car1'].unique()[0]
                    car2 = feature_imp2['car2'].unique()[0]
                    
                    comparator1_category = feature_imp1_selected['category'].unique()[0]
                    comparator1_effect = feature_imp1_selected['effect'].unique()[0]
                
                    comparator2_category = feature_imp2_selected['category'].unique()[0]
                    comparator2_effect = feature_imp2_selected['effect'].unique()[0]
                    
                    feature_imp1_selected = feature_imp1_selected[['package_option', 'category', 'effect', 'jdp_feature_code', 'jdp_feature_name']]
                    feature_imp2_selected = feature_imp2_selected[['package_option', 'category', 'effect', 'jdp_feature_code', 'jdp_feature_name']]
                    
                    feature_imp1_selected.columns = ['comparator1_package_option', 'comparator1_category', 'comparator1_effect','comparator1_feature_code', 'comparator1_feature_name']
                    feature_imp2_selected.columns = ['comparator2_package_option', 'comparator2_category', 'comparator2_effect', 'comparator2_feature_code', 'comparator2_feature_name']
                    
                    feature_imp_selected = pd.merge(feature_imp1_selected, feature_imp2_selected, left_on = 'comparator1_feature_code',
                                               right_on='comparator2_feature_code', how='outer')
                    
                    feature_imp_selected.drop(['comparator1_package_option', 'comparator2_package_option', 'comparator1_feature_code', 'comparator2_feature_code', 'comparator1_category', 'comparator1_effect', 'comparator2_category', 'comparator2_effect'], axis=1, inplace=True)
        
                    comparator1_value = comparator1_pkg + ' (' + comparator1_category + ', ' + comparator1_effect + ')'
                    comparator2_value = comparator2_pkg + ' (' + comparator2_category + ', ' + comparator2_effect + ')'
                    
                    feature_imp_selected.columns = ['car1', 'car2']
            
                
                    pkg_features_table = dash_table.DataTable(
                                                id='pkg_features_table_result',
                                                style_data={
                                                'whiteSpace': 'normal',
                                                'height': 'auto'
                                            },
                                                style_cell={'textAlign': 'center'},
                                                #columns=[{"name": i, "id": i} for i in feature_imp_selected.columns],
                                                columns = [{'name': [car1, comparator1_value], 'id': 'car1'},
                                                            {'name': [car2, comparator2_value], 'id': 'car2'}],
                                                data=feature_imp_selected.to_dict('records'),
                                            style_data_conditional = [{'if': {"row_index": x},'backgroundColor': 'cyan'} for x in feature_imp_selected[feature_imp_selected['car1'] == feature_imp_selected['car2']].index.tolist()],  
                                            style_header={
                                                'backgroundColor': 'rgb(230, 230, 230)',
                                                'fontWeight': 'bold',
                                                'whiteSpace': 'normal',
                                                'height': 'auto'
                                            },
                                            style_header_conditional=[
                                            {
                                                'if': {
                                                    'column_id': 'car1',
                                                },
                                                'color': effect_color[comparator1_effect],
                                            },
                                            {
                                                'if': {
                                                    'column_id': 'car2',
                                                },
                                                'color': effect_color[comparator2_effect],
                                            },
                                        ]
                                                
                                            )
                    
            
                    return pkg_features_table
#            

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
