from dash import Dash,Input, Output, dcc, html,callback_context, State,dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import psycopg2
import numpy as np
from Utils import process_dx

external_stylesheets = ['styles.css']
app = Dash(__name__, title="ECG Analysis",external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
dataset_path = r'..\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\\'


Nav = html.Nav(id='navbar', children=[
    html.Div('Expolatory Data Analysis of PhysioNet\'s 12 Lead ECG Data Regarding Arrhythmia', id='nav-title'),
    html.Div([
        html.A('Home', id='home-link', href='#home', className='nav-link'),
        html.A('Age', id='age-link', href='#age', className='nav-link'),
        html.A('Gender', id='gender-link', href='#gender', className='nav-link'),
        html.A('Diagonosis', id='diagonosis-link', href='#diagonosis', className='nav-link'),
        html.A('PCA', id='pca-link', href='#pca', className='nav-link'),
        html.A('Analysis', id='analysis-link', href='#analysis', className='nav-link')
    ], id='nav-links')
])

app.layout = html.Div([
    Nav,
    html.Div(id='content')
])

@app.callback(
    Output('content', 'children'),
    [Input('home-link', 'n_clicks'),
     Input('age-link', 'n_clicks'),
     Input('gender-link', 'n_clicks'),
     Input('diagonosis-link', 'n_clicks'),
     Input('pca-link', 'n_clicks'),
     Input('analysis-link', 'n_clicks')]
)
def display_content(home_clicks, age_clicks, gender_clicks, diagonosis_clicks ,pca_clicks, analysis_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return Home_section
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'home-link':
            return Home_section
        elif button_id == 'age-link':
            return Age_section
        elif button_id == 'gender-link':
            return Gender_section
        elif button_id == 'pca-link':
            return PCA_section
        elif button_id == 'analysis-link':
            return Analysis_section
        elif button_id == 'diagonosis-link':
            return DX_section

def fetch_data_table(page_number, page_size):
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT id,age,gender,dx,hr FROM ecg ORDER BY id OFFSET %s LIMIT %s", ((page_number - 1) * page_size, page_size))
    data = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    cur.close()
    conn.close()
    
    return pd.DataFrame(data, columns=columns)

# Function to fetch record data from the database
def get_record(id):
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT id,p_signal,n_signal,age,gender,dx from ECG where id = %s", (id,))
    record = cur.fetchall()
    cur.close()
    conn.close()
    
    data = [np.frombuffer(record[0][1], dtype=np.float64).reshape(-1, 12), np.frombuffer(record[0][2], dtype=np.float64).reshape(-1, 12)]
    patient_info = {
        'id': record[0][0],
        'age': record[0][3],
        'gender': record[0][4],
        'dx': record[0][5]
    }
    
    return data, patient_info

# Function to visualize ECG data
def visualize_data(record1, record2, patient_info):
    figs = []
    leads = ["I", "II", "III", "avR", "avF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    colors = ['orange', 'red']
    
    for i, lead in enumerate(leads):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(record1[:, i]))), y=record1[:, i], mode='lines', name=f"{lead} (Original Signal)", line=dict(color=colors[0])))
        fig.add_trace(go.Scatter(x=list(range(len(record2[:, i]))), y=record2[:, i], mode='lines', name=f"{lead} (Filtered Signal)", line=dict(color=colors[1])))
        fig.update_layout(
            title=f"Lead {lead}",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            paper_bgcolor="#3b3b3b",  # Set paper color
            font=dict(color="#fec036"),  # Set title and axis labels color
            legend=dict(
                x=0,
                y=1,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="white"
                ),
                bgcolor="#3b3b3b",
                bordercolor="gray",
                borderwidth=1
            )
        )
        
        figs.append(fig)
        
      
    figs[0].update_layout(
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Patient ID: {patient_info['id']}, Age: {patient_info['age']}, Gender: {patient_info['gender']}, Diagnosis: {process_dx(patient_info['dx'])}",
                showarrow=False,
                font=dict(color="#fec036")
            )
        ]
    )
         
    return figs

# Define the layout of the dashboard
Home_section = html.Div([
    html.H1("ECG Leads Visualization", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px"}),
    html.Div([
        dash_table.DataTable(
            id='data-table',
            columns=[
                {"name": col, "id": col} for col in ['id', 'age', 'gender', 'dx', 'hr']
            ],
            page_current=0,
            page_size=20,
            page_action='custom',
            style_table={
                'border': '0px solid #fec036',  # Border color and thickness
                'borderCollapse': 'collapse',
                'fontWeight': 'bold'
            },
            style_cell={
                'backgroundColor': 'rgba(43, 43, 43, 0.7)',  # Background color of cells
                'color': 'white',  # Text color of cells
                'border': '0px solid #fec036',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgba(255,255,255,0.9)',
                    'color': '#1e1e1e'
                },
                {
                    'if': {'row_index': 'even'},
                    'backgroundColor': '#7b7b7b',
                    'color': 'white'
                },
                {
                    'if': {'state': 'active'},  # Conditional formatting for the active cell
                    'backgroundColor': '#fec034',  # Background color of active cell
                    'color': 'white',  # Text color of active cell
                    'fontWeight': 'bold'  # Bold text in active cell
                },{
                    'if': {'filter_query': '{id} = None'},  # Conditional formatting for column header rows
                    'color': 'white'  # Text color of column header rows
                },
            ]
        )
    ], style={'backgroundColor': '#3b3b3b',"padding":"30px"}),
    html.Div([
        dcc.Input(id='record-id', type='text', placeholder='Enter ID: JSXXXXX e.g JS12345', className="text-field"),
        html.Button('Visualize', id='visualize-btn', n_clicks=0, className="button"),
    ], style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px"}),
    html.Div(id='graphs-container'),
], id='home', style={"color": "white"})

# Callback to update the patient information and graphs
@app.callback(
    Output('graphs-container', 'children'),
    [Input('visualize-btn', 'n_clicks')],
    [State('record-id', 'value')]
)
def update_graph(n_clicks, record_id):
    if n_clicks > 0:
        record_data, patient_info = get_record(record_id)
        figs = visualize_data(record_data[0], record_data[1], patient_info)
        graph_elements = []
        for i, fig in enumerate(figs):
            graph_elements.append(
                dcc.Graph(
                    id=f'ecg-graph-{i}',
                    figure=fig
                )
            )
        return graph_elements
    else:
        return []

@app.callback(
    Output('data-table', 'data'),
    [Input('data-table', 'page_current')],
    [State('data-table', 'page_size')]
)
def update_table(page_current, page_size):
    data = fetch_data_table(page_current + 1, page_size)  # Pages are 0-indexed, adjust to 1-indexed for SQL OFFSET
    return data.to_dict('records')

# Callback to store the clicked ID value in the dcc.Input component
@app.callback(
    Output('record-id', 'value'),
    [Input('data-table', 'active_cell')]
)
def store_selected_id(active_cell):
    if active_cell:
        row = active_cell['row']
        column_id = active_cell['column_id']
        if column_id == 'id':
            return active_cell['row_id']
    return ''


# Function to connect to the database and fetch data
def fetch_age_data():
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin"
    )

    # Create a cursor
    cur = conn.cursor()
    cur.execute("SELECT age,dxcount from ECG")
    data = cur.fetchall()
    cur.close()
    conn.close()

    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(data, columns=['age','dxcount'], dtype='int64')
    return df

# Layout for the Age section
Age_section = html.Div([
    html.H1("Age Data Distributions", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px", "color": "white"}),
     html.Div([
        dcc.Graph(id='graph-age',style={"display": "flex", "flex-basis": "65%"}),
        dcc.Graph(id='graph-pie',style={"display": "flex", "flex-basis": "30%"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),

    html.Div([
        dcc.Graph(id='graph-box',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
     html.Div([
        dcc.Graph(id='graph-scatter',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
])

# Callback to update the graphs
@app.callback(
    [Output('graph-age', 'figure'),
     Output('graph-box', 'figure'),
     Output('graph-pie', 'figure'), Output('graph-scatter', 'figure')],
    [Input('graph-age', 'relayoutData')]
)
def update_graphs(relayoutData):
    df = fetch_age_data()
    
    # Define age groups
    bins = [0,10, 20, 30, 40, 50, 60, 70, 80, 90]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90']
    
    # Bin ages into groups
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    
    # Count occurrences of each age group
    age_group_counts = df['age_group'].value_counts().sort_index()
    
    # Define colors for each bar
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'gray']
    
    # Create bar chart trace with different colors for each bar
    bar_data = []
    for i, (age_group, count) in enumerate(age_group_counts.items()):
        trace = go.Bar(
            x=[age_group],
            y=[count],
            marker=dict(color=colors[i]),  # Set bar color
            name=age_group  # Set legend name
        )
        bar_data.append(trace)

    layout_bar = go.Layout( 
        xaxis=dict(title='Age Group'),  
        yaxis=dict(title='Count'), 
        barmode='group' ,
        title=f"Age Distribution in Term of Numbers",
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )

    fig_bar = go.Figure(data=bar_data, layout=layout_bar)

    # Box plot
    box_fig = px.box(df, x='dxcount', y='age', title='Box Plot: Age vs Diagnosed Diseases Count',
                     labels={'dxcount': 'Diagnosed Diseases Count', 'age': 'Age'})
    # box_fig.update_traces(marker=dict(color='white'), line=dict(color='#8b8b8b'), marker_outliercolor='white')
        # boxpoints='all',
        # jiter=0.3
    # box_fig.update_layout(
    #     # paper_bgcolor="#3b3b3b",  # Set paper color
    #     # plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
    #     # font=dict(color="#fec036"),  # Set title and axis labels color 
    #     # xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
    #     # yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
    #     # xaxis_gridwidth=1,  # Set x-axis grid line width
    #     # yaxis_gridwidth=2,  # Set y-axis grid line width
    # )



    # Pie chart
    count_by_category = df['age_group'].value_counts().reset_index()
    count_by_category.columns = ['Age Group', 'Count']
    pie_fig = px.pie(count_by_category, values='Count', names='Age Group',
                     title='Pie Chart: Age Distribution', hole=0.4)
    pie_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )
    
    fig_scatter = px.scatter(df, x='age', y='dxcount', title='Age vs. Diagnosis Count', trendline="ols")
    fig_scatter.update_traces(marker=dict(color='white'))

# Update trendline color to red
    fig_scatter.update_traces(line=dict(color='white', width=2))
    fig_scatter.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )

    return fig_bar, box_fig, pie_fig, fig_scatter

def fetch_gender_data():
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin"
    )

    # Create a cursor
    cur = conn.cursor()
    cur.execute("SELECT gender,dxcount from ECG")
    data = cur.fetchall()
    cur.close()
    conn.close()

    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(data, columns=['gender','dxcount'])
    return df

# Layout for the Age section
Gender_section = html.Div([
    html.H1("Gender Data Distributions", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px", "color": "white"}),
     html.Div([
        dcc.Graph(id='graph-gender-bar',style={"display": "flex", "flex-basis": "65%"}),
        dcc.Graph(id='graph-gender-pie',style={"display": "flex", "flex-basis": "30%"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),

    html.Div([
        dcc.Graph(id='graph-gender-box',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
    html.Div([
        dcc.Graph(id='graph-gender-kde',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
])

# Callback to update the graphs
@app.callback(
    [Output('graph-gender-bar', 'figure'),
     Output('graph-gender-box', 'figure'),
     Output('graph-gender-pie', 'figure'),
     Output('graph-gender-kde', 'figure')],
    [Input('graph-gender-bar', 'relayoutData')]
)
def update__gender_graphs(relayoutData):
    df = fetch_gender_data()
    
    df_grouped = df.groupby(['gender', 'dxcount']).size().reset_index(name='count')
    total_counts = df['gender'].value_counts().to_dict()
    df_grouped['percentage'] = df_grouped.apply(lambda row: row['count'] / total_counts[row['gender']] * 100, axis=1)
    
    bar_fig = px.bar(df_grouped, x='dxcount', y='percentage', color='gender', barmode='group')

    bar_fig.update_layout(
        xaxis=dict(title='Diagnosis Count'),  
        yaxis=dict(title='Gender Percentage'), 
        title=f"Percentage of Diagnosis Count by Gender",
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )
    kde_fig = px.histogram(df, x="dxcount", color="gender", marginal="rug")

    # Update layout
    kde_fig.update_layout(
        xaxis=dict(title='Diagnosis Count'),  
        yaxis=dict(title='Count'), 
        title="Kernel Density Estimation (KDE) Plot of Diagnosis Count by Gender",
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
        barmode="overlay"
    )
    kde_fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.5)')  # For y-axis grid
    kde_fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.5)')  # For x-axis grid

    # Box plot
    box_fig = px.box(df, x='gender', y='dxcount', title='Box Plot: Age vs Diagnosed Diseases Count',
                     labels={'dxcount': 'Diagnosed Diseases Count', 'gender': 'Age'})
    box_fig.update_traces(marker=dict(color='#fec036'), line=dict(color='white'), marker_outliercolor='red')
        # boxpoints='all',
        # jiter=0.3
    box_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )

    pie_fig = px.pie(df, names='gender',
                     title='Gender Distribution Among Data', hole=0.4)
    pie_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )
    
    return bar_fig, box_fig, pie_fig, kde_fig

def fetch_dx_data():
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin"
    )

    # Create a cursor
    cur = conn.cursor()
    cur.execute("SELECT id,age,gender,dxcount,dx,dxname FROM ecg")
    data = cur.fetchall()
    cur.close()
    conn.close()

    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(data, columns=['id','age','gender','dxcount','dx','dxname'])
    return df

DX_section = html.Div([
    html.H1("Diagnosis Data Distributions", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px", "color": "white"}),
     html.Div([
        dcc.Graph(id='graph-dx-hist')
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),
     html.Div([
        dcc.Graph(id='graph-dx-topbar',style={"display": "flex", "flex-basis": "65%"}),
        dcc.Graph(id='graph-dx-heat',style={"display": "flex", "flex-basis": "30%"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),
    html.Div([
        dcc.Graph(id='graph-dx-genderageheat',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
    html.Div([
        dcc.Graph(id='graph-dx-covmat',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
])

# Callback to update the graphs
@app.callback(
    [Output('graph-dx-hist', 'figure'),
     Output('graph-dx-topbar', 'figure'),
     Output('graph-dx-heat', 'figure'),
     Output('graph-dx-genderageheat', 'figure'),
     Output('graph-dx-covmat', 'figure')],
    [Input('graph-dx-hist', 'relayoutData')]
)
def update__gender_graphs(relayoutData):
    df = fetch_dx_data()
    df['dx'] = df['dx'].str.split(',')
    df['dxname']=df['dxname'].str.split(',')
    
    diagnosis_codes = sorted(set(code for sublist in df['dxname'] for code in sublist))
    co_occurrence_matrix = pd.DataFrame(0, index=diagnosis_codes, columns=diagnosis_codes)

    for codes in df['dxname']:
        for code1 in codes:
            for code2 in codes:
                co_occurrence_matrix.loc[code1, code2] += 1

    # Convert the index and columns to lists
    co_occurrence_matrix.index = co_occurrence_matrix.index.tolist()
    co_occurrence_matrix.columns = co_occurrence_matrix.columns.tolist()

    # 1. Diagnosis Count Distribution (Histogram)
    fig_hist = px.histogram(df, x='dxcount', color_discrete_sequence=['#fec036'])
    fig_hist.update_layout(
        xaxis=dict(title='Diagnosis Count'),  
        yaxis=dict(title='Count'),
        title=f"Diagnosis Count Distribution",
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
        barmode="overlay"
    )
    # 2. Top Diagnoses (Bar Chart)
    top_diagnoses = df['dxname'].explode().value_counts().nlargest(10)
    fig_top_diagnoses = px.bar(x=top_diagnoses.index, y=top_diagnoses.values, color=top_diagnoses.index)
    fig_top_diagnoses.update_layout(
        xaxis=dict(title='Diagnosis Count'),  
        yaxis=dict(title='Count'), 
        title=f"Top Diagnoses",
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
        barmode="overlay"
    )
    
    fig_diagnosis_co_occurrence = go.Figure(data=go.Heatmap(z=np.array(co_occurrence_matrix),
                                                       x=co_occurrence_matrix.index,
                                                       y=co_occurrence_matrix.columns))
    fig_diagnosis_co_occurrence.update_layout(
        title='Heatmap of Diagnosis Co-occurrence',
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width)
    )
        
    heatmap_data = df.pivot_table(index='age', columns='gender', values='dx', aggfunc='mean')

    # Create the heatmap
    fig_heatmap = px.imshow(heatmap_data,
                            labels=dict(x="Gender", y="Age", color="Diagnosis Count"),
                            x=heatmap_data.columns,
                            y=heatmap_data.index,
                            title='Heatmap of Diagnosis Count by Age and Gender')
    fig_heatmap.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width)
    )
    
    diagnoses_sets = [set(diagnosis.split(',')) for diagnosis in top_diagnoses.index]

    # Create a matrix representing the overlap between sets
    overlap_matrix = np.zeros((len(diagnoses_sets), len(diagnoses_sets)))
    for i, set1 in enumerate(diagnoses_sets):
        for j, set2 in enumerate(diagnoses_sets):
            overlap_matrix[i, j] = len(set1.intersection(set2))

    # Create a heatmap
    fig = go.Figure(data=go.Heatmap(
        z=overlap_matrix,
        x=top_diagnoses.index,
        y=top_diagnoses.index,
        colorscale='Viridis',
        colorbar=dict(title='Overlap Count')
        
    ))

    # Update layout
    fig.update_layout(
        title='Overlap between Top 10 Diagnoses',
        xaxis=dict(title='Diagnosis'),
        yaxis=dict(title='Diagnosis'),
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width)
    )
        
    return fig_hist, fig_top_diagnoses, fig, fig_heatmap, fig_diagnosis_co_occurrence

PCA_section = html.Div([
    html.H1("Principle Component Analysis", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px", "color": "white"}),
    html.Div([
        dcc.Input(id='pca-id', type='text', placeholder='Enter ID: JSXXXXX e.g JS12345', className="text-field"),
        html.Button('Visualize', id='visualizebtn', n_clicks=0, className="button"),
    ], style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px"}),
    
    html.Div( id='graph-pca-hist',style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"})
])

@app.callback(
    Output('graph-pca-hist', 'children'),
    [Input('visualizebtn', 'n_clicks')],
    [State('pca-id', 'value')]
)
def update_graph(n_clicks, record_id):
    if n_clicks > 0:
        record_data= get_pca_record(record_id)
        figs = visualize_pca_data(record_data[0], record_data[1])
        graph_elements = []
        for i, fig in enumerate(figs):
            graph_elements.append(
                dcc.Graph(
                    id=f'ecg-graph-{i}',
                    figure=fig
                )
            )
        return graph_elements
    else:
        return []

def get_pca_record(id):
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin",
        port="5432"
    )
    cur = conn.cursor()
    cur.execute("SELECT id,n_signal,pca from ECG where id = %s", (id,))
    record = cur.fetchall()
    cur.close()
    conn.close()
    
    data = [np.frombuffer(record[0][1], dtype=np.float64).reshape(-1, 12), np.frombuffer(record[0][2], dtype=np.float64).reshape(-1, 1)]
    
    
    return data

def visualize_pca_data(record1, record2):
    figs = []
    
    histogram_trace = go.Histogram(x=record1.flatten(), nbinsx=20,marker=dict(color="#fec036"))

    layout = go.Layout(
        title='Distribution of Original Data (12 Component)',
        xaxis=dict(title='Transformed Data Values'),
        yaxis=dict(title='Frequency'),
        bargap=0.1,  # Gap between bars
        bargroupgap=0.1,  # Gap between groups of bars
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Shape: {record1.shape}",
                showarrow=False,
                font=dict(color="#fec036")
            )
        ]
    )

    # Combine trace and layout
    fig = go.Figure(data=[histogram_trace], layout=layout)
      
    figs.append(fig)
    
    histogram_trace = go.Histogram(x=record2.flatten(), nbinsx=20,marker=dict(color="#fec036"))

    layout = go.Layout(
        title='Distribution of Transformed Data (1 Component)',
        xaxis=dict(title='Transformed Data Values'),
        yaxis=dict(title='Frequency'),
        bargap=0.1,  # Gap between bars
        bargroupgap=0.1,  # Gap between groups of bars
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2, 
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Shape: {record2.shape}",
                showarrow=False,
                font=dict(color="#fec036")
            )
        ]
    )

    # Combine trace and layout
    fig = go.Figure(data=[histogram_trace], layout=layout)
    figs.append(fig)    
    return figs


def fetch_hr_data():
    conn = psycopg2.connect(
        host="localhost",
        database="EDA",
        user="postgres",
        password="admin"
    )

    # Create a cursor
    cur = conn.cursor()
    cur.execute("SELECT id,age,gender,dxcount,dx,hr,dxname FROM ecg")
    data = cur.fetchall()
    cur.close()
    conn.close()

    # Convert the fetched data into a DataFrame
    df = pd.DataFrame(data, columns=['id','age','gender','dxcount','dx','hr','dxname'])
    return df

Analysis_section = html.Div([
    html.H1("Data Analysis Based Upen Heart Rate and Arrhythmia", style={"display": "flex", "flex-direction": "column", "align-items": "center", "margin-bottom": "15px", "font-size": "50px", "color": "white"}),
     dcc.Graph(id='graph-hr-hist'),
     html.Div([
        dcc.Graph(id='graph-hr-box',style={"display": "flex", "flex-basis": "65%"}),
        dcc.Graph(id='graph-hr-pie',style={"display": "flex", "flex-basis": "30%"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),
    html.Div([
    
        dcc.Graph(id='graph-hr-countratiopie',style={"display": "flex", "flex-basis": "40%"}),
        dcc.Graph(id='graph-hr-dieaseratiopie',style={"display": "flex", "flex-basis": "40%"})
    ], style={"display": "flex", "flex-direction": "row", "justify-content": "space-around", "margin-top": "30px"}),
    html.Div([
        dcc.Graph(id='graph-hr-heartbox',style={"height":"90vh"})
    ], style={"margin-top": "30px"}),
])

# Callback to update the graphs
@app.callback(
    [Output('graph-hr-hist', 'figure'),
     Output('graph-hr-box', 'figure'),
     Output('graph-hr-pie', 'figure'),
     Output('graph-hr-countratiopie', 'figure'),
     Output('graph-hr-dieaseratiopie', 'figure'),
     Output('graph-hr-heartbox', 'figure')],
    [Input('graph-hr-heartbox', 'relayoutData')]
)
def update__gender_graphs(relayoutData):
    df = pd.DataFrame(fetch_hr_data())

    # Convert 'dx' column to lists
    df['dx'] = df['dx'].apply(lambda x: x.split(','))
    df['dxname'] = df['dxname'].apply(lambda x: x.split(','))

    # Define the specified diseases
    specified_diseases = ['1AVB', '2AVB', '2AVB1', '2AVB2', '3AVB', 'ABI', 'APB', 'AVB', 'JEB', 'JPT', 'LBBB', 'RBBB', 'SB', 'SR', 'AFIB', 'AF', 'SVT', 'AT', 'AVNRT', 'AVRT']

    # Calculate the number of patients having one or more of the specified diseases
    df['has_specified_disease'] = df['dxname'].apply(lambda x: any(disease in x for disease in specified_diseases))
    disease_ratio_fig = px.pie(df, names='has_specified_disease', title='Ratio of Patients with Specified Diseases',hole=0.5)
    
    disease_ratio_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )
    

    # Calculate the average heart rate for patients with and without the specified diseases
    heart_rate_effect_fig = px.box(df, x='has_specified_disease', y='hr', title='Effect on Heart Rate by Specified Diseases', labels={'has_specified_disease': 'Has Specified Disease', 'hr': 'Heart Rate'})

    heart_rate_effect_fig.update_traces(marker=dict(color='white'), line=dict(color='#8b8b8b'), marker_outliercolor='white')
        # boxpoints='all',
        # jiter=0.3
    heart_rate_effect_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )
    
    # Calculate the ratio of men to women for each specified disease
    gender_ratio_df = df.explode('dxname')[['dxname', 'gender']]
    gender_ratio_df = gender_ratio_df[gender_ratio_df['dxname'].isin(specified_diseases)]
    gender_ratio_fig = px.histogram(gender_ratio_df, x='dxname', color='gender', histfunc='count')
    
    gender_ratio_fig.update_layout(
        xaxis=dict(title='Gender'),  
        yaxis=dict(title='Disease'),
        title=f'Ratio of Men to Women for Specified Diseases',
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
        barmode="group"
    )

    df['num_specified_diseases'] = df['dxname'].apply(lambda x: sum(disease in x for disease in specified_diseases))

    df['num_specified_diseases_class'] = df['num_specified_diseases'].apply(lambda x: '3 or more' if x > 2 else str(x))
    pie_chart_fig = px.pie(df, names='num_specified_diseases_class', title='Breakdown of Number of Diseases per Patient',hole=0.5)
    pie_chart_fig.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )

    df_specified_diseases = df[df['dxname'].apply(lambda x: any(disease in x for disease in specified_diseases))]
    # Create a column counting the number of specified diseases for each patient
    df_specified_diseases['num_specified_diseases'] = df_specified_diseases['dxname'].apply(lambda x: min(len(x), 4))

    
    # Create a pie chart showing the breakdown of patients based on the number of specified diseases they have
    pie_chart_specified_diseases = px.pie(df_specified_diseases, names='num_specified_diseases', 
                                        title='Breakdown of Patients by Number of Specified Diseases',
                                        labels={'num_specified_diseases': 'Number of Specified Diseases'},
                                        hole=0.5)
    pie_chart_specified_diseases.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )

    # Box plot for heart rate against specified diseases
    box_plot_specified_diseases = px.box(df_specified_diseases.explode('dxname'), x='dxname', y='hr', 
                                        title='Heart Rate vs. Specified Diseases',
                                        labels={'dx': 'Disease Name', 'hr': 'Heart Rate'})

    box_plot_specified_diseases.update_traces(marker=dict(color='white'), line=dict(color='#8b8b8b'), marker_outliercolor='white')
        # boxpoints='all',
        # jiter=0.3
    box_plot_specified_diseases.update_layout(
        paper_bgcolor="#3b3b3b",  # Set paper color
        plot_bgcolor="rgba(0,0,0,0)",  # Set plot background color
        font=dict(color="#fec036"),  # Set title and axis labels color 
        xaxis_gridcolor='rgba(254, 192, 54, 0.8)',  # Set x-axis grid line color
        yaxis_gridcolor='rgba(254, 192, 54, 0.7)',  # Set y-axis grid line color
        xaxis_gridwidth=1,  # Set x-axis grid line width
        yaxis_gridwidth=2,  # Set y-axis grid line width
    )


    return gender_ratio_fig, heart_rate_effect_fig, disease_ratio_fig, pie_chart_specified_diseases, pie_chart_fig, box_plot_specified_diseases



if __name__ == '__main__':
    app.run_server(debug=True)
