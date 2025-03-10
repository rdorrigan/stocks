from dash import Dash, dcc, html, Input, Output, State, Patch, no_update, clientside_callback, callback
from dash.dash_table import DataTable, FormatTemplate
from dash_bootstrap_templates import load_figure_template
import plotly.io as pio
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import timedelta, datetime
from data.data import fetch_stock_data, get_predictions
from data.db import query_db

template_themes = ['vizro', 'vizro_dark']
load_figure_template(template_themes)
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.2.4/dbc.min.css"

def get_style_sheets():
    return [dbc.themes.BOOTSTRAP, dbc.themes.MINTY, dbc.icons.FONT_AWESOME, dbc_css]
# Components
children = []
def create_header():
    '''
    Dashboard Header
    '''
    return html.H1("Stock Price Dashboard", style={'textAlign': 'center'})
def create_color_mode_switch():
    '''
    Light and Dark mode switch
    '''
    return html.Span([dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch(id="color-mode-switch", value=False,
                   className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch")])

def create_stock_selector():
    '''
    Stock dropdown and input box
    '''

    odf = query_db('select ticker,label from tickers order by MarketCap desc')
    odf = odf.rename(columns={'Ticker':'value','Label':'label'})
    print(odf)
    options = odf.to_dict('records')
    print(options)
    del odf

    drop_down_header = html.H4("Stock Selector", style={'textAlign': 'center', 'margin': '10px'})
    drop_down_input = html.Div([
        dcc.Dropdown(
            id="stock-selector",
            options=options,
            value="AAPL",
            clearable=False,
            style={'width': '50%', 'margin': 'auto'},
            multi=False
        ),
        dcc.Input(
            id='input-box',
            type='text',
            debounce=True,
            placeholder='Enter other stock NASDAQ symbol',
            value=None,
            style={'width': '20%', 'margin': 'auto'}
        ),
        dcc.Store(id="error-store"),
        dbc.Alert(id="error-alert", color="danger", is_open=False, dismissable=True)
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}, className='dbc')
    
    return html.Div([drop_down_header, drop_down_input])
def create_stock_card(summary_date='',price="", pct=""):
    '''
    Display latest closing price and the percentage change from the first closing price
    '''
    if not isinstance(price, str):
        price = f'${price}'
    if not isinstance(pct, str):
        pct = f'{pct:.2%}'
    print(price,pct)
    summary_header = html.H4(f"Summary as of {summary_date}", style={
                      'textAlign': 'center', 'margin': '10px'})
    summary_price = html.H5(price, id='summary-price',
                            style={'textAlign': 'center', 'margin': '10px'})
    summary_pct = html.H5(pct, 'summary-pct',
                          style={'textAlign': 'center', 'margin': '10px'})
    summaries = html.Div([summary_price,summary_pct],style={
        'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'})
    return html.Div([summary_header, summaries], id='summary-card', style={
        'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
def create_model_selector():
    '''
    Filter selection of predictive models
    '''
    model_header = html.H4("Predictive Model Selector", style={'textAlign': 'center', 'margin': 'auto',
                                                               'display': 'flex', 'flexDirection': 'column', 'alignItems': 'left'})
    model_selector = dcc.RadioItems(
        id="model-selector",
        options=[
            {"label": "XGBoost", "value": "xgboost"},
            {"label": "ARIMA", "value": "arima"},
            {"label": "LightGBM", "value": "lightgbm"},
            {"label": "LinearRegression", "value": "linearregression"}
        ],
        value="xgboost",
        labelStyle={'display': 'inline-block', 'margin': '10px'}
    )
    return html.Div([model_header, model_selector], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'left'})

def create_date_selector():
    '''
    Available date filters
    '''
    period_labels = ['5D', '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y']
    date_period_selector = dcc.RadioItems(
        id="date-period-selector",
        options=[{"label": l, "value": l} for l in period_labels],
        value=period_labels[-1],
        labelStyle={'display': 'inline-block', 'margin': '10px'}
    )
    date_header = html.H4("Date Filters", style={'textAlign': 'center', 'margin': '10px'})
    return html.Div([date_header, date_period_selector], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})

def selector_layout():
    return html.Div([create_model_selector(), create_date_selector()], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'left'})

def create_price_graph():
    '''
    Price graph holder
    '''
    return dcc.Graph(id="stock-price-chart")

def create_file_download_section():
    '''
    DataTable File download options and button
    '''
    file_type_header = html.H4("Choose download file type.", style={'textAlign': 'left'})
    file_type_selector = dcc.RadioItems(
        id="file-type-selector",
        options=[{"label": "Excel file", "value": "xlsx"},
                 {"label": "CSV file", "value": "csv"}],
        value='csv',
        labelStyle={'display': 'inline-block', 'margin': '10px'}
    )
    download_button = html.Button("Download Data", id='download-button', style={"marginTop": 20})
    download_component = dcc.Download(id='download')
    file_div = html.Div([file_type_header, file_type_selector], id='file-div', style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'})
    download_div = html.Div([download_button, download_component], id='download-div', style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'})
    return html.Div([file_div, download_div], id='file-download-div', style={'display': 'flex', 'flex-direction': 'row', 'margin': '10px'})
def data_table_style(dark) -> dict:
    '''
    DataTable style params for light and dark mode
    '''
    if not dark:
        return dict(style_data={
            'color': 'black',
            'backgroundColor': 'white'
        },
            style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(220, 220, 220)',
            }
        ],
            style_header={
            'backgroundColor': 'rgb(210, 210, 210)',
            'color': 'black',
            'fontWeight': 'bold'
        })

    return dict(style_header={
        'backgroundColor': 'rgb(30, 30, 30)',
        'color': 'white'
    },
        style_data_conditional=[],
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
    })
def create_data_table():
    ''''
    DataTable displaying price information for the given dates
    '''
    money = FormatTemplate.money(2)
    percent = FormatTemplate.percentage(2, True)
    return DataTable(id='data-table', page_size=10, columns=[
        {'name': 'Date', 'id': 'Date', 'type': 'datetime'},
        {'name': 'Price', 'id': 'Close', 'format': money, "type": 'numeric'},
        {'name': '50 Day MA', 'id': '50_MA', 'format': money, "type": 'numeric'},
        {'name': '200 Day MA', 'id': '200_MA', 'format': money, "type": 'numeric'},
        {'name': 'Daily Return', 'id': 'Daily_Return', 'format': percent, "type": 'numeric'},
        {'name': 'Volatility', 'id': 'Volatility', 'format': percent, "type": 'numeric'}
    ], sort_action="native", filter_action='native', style_table={"overflowX": "auto"},
    **data_table_style(False))



def create_layout():
    # List of children components that will display in order
    children = [
        create_color_mode_switch(),
        create_header(),
        create_stock_selector(),
        create_stock_card(),
        # create_model_selector(),
        # create_date_selector(),
        selector_layout(),
        create_price_graph(),
        create_file_download_section(),
        create_data_table()
    ]
    # Layout
    return html.Div(children=children, id='layout', className="dbc")


@callback(
    Output("error-alert", "children"),
    Output("error-alert", "is_open"),
    Input("error-store", "data"),
)
def display_error(error_message):
    '''
    Display error message if the symbol in the input box was invalid
    '''
    if error_message:
        return error_message, True
    return "", False  # Hide alert if no error


# Remove text from input box when a new stock has been selected
@callback(Output('input-box', 'value'), Input("stock-selector", "value")
                   )
def clean_input_box(stock_selected):
    '''
    Erase the input box when another stock is selected from the dropdown
    '''
    return ""
# Callback to update graphs


def period_to_date_range(date_filter):
    '''
    Returns a tuple of min and max datetimes from the date_range for filter stock_data
    '''
    if date_filter[1] == 'Y':
        delta = int(date_filter[0]) * 365
        return (datetime.today() - timedelta(days=delta)), datetime.today()
    elif date_filter == '1M':
        delta = 30
        return (datetime.today() - timedelta(days=delta)), datetime.today()
    date_range = pd.date_range(end=datetime.today(), periods=int(
        date_filter[0]), freq=date_filter[1], normalize=True)
    return date_range.min().to_pydatetime(), date_range.max().to_pydatetime()


@callback(
    [
        Output('data-table', "data"),
        Output("stock-price-chart", "figure", allow_duplicate=True),
        Output('summary-card', 'children'),
        Output("error-store", "data")  # Store error messages,
    ],
    [Input("stock-selector", "value"),
     Input('input-box', 'value'),
     Input("model-selector", "value"),
     Input("date-period-selector", "value")],
    State("color-mode-switch", "value"),
    prevent_initial_call=True
)
def update_graphs(selected_stock, input_value, model_type, date_filter, switch_on):
    '''
    Update line graphs with stock data and model predictions
    '''
    selected_stock = selected_stock if not input_value else input_value.strip().upper()

    # fetch stock_data saving to csv avoiding additional API calls
    stock_data = fetch_stock_data(selected_stock)
    stock_data_cols = ['Date', 'Close', '50_MA',
                       '200_MA', 'Daily_Return', 'Volatility']
    if stock_data is None or stock_data.empty:
        # dash.no_update
        return [{c: '' for c in stock_data_cols}], go.Figure(), create_stock_card(), f"Invalid stock symbol: {input_value}"

    # Train model or get cached model predictions
    future_dates, future_prices = get_predictions(
        selected_stock, stock_data, model_type)
    stock_data = stock_data[stock_data_cols]
    stock_data[stock_data_cols[2:]] = stock_data[stock_data_cols[2:]].round(2)
    if date_filter not in ('5Y', None):
        date_range = period_to_date_range(date_filter)
        tdf = stock_data.loc[stock_data['Date'].between(
            *date_range, inclusive='both'), :]
        if not tdf.empty:
            stock_data = tdf.copy()
            del tdf
    summary_date = stock_data['Date'].max()
    summary_price = stock_data.loc[stock_data['Date']
                                   == summary_date, 'Close']#.iloc[0]
    print(summary_price)
    summary_price = summary_price.values[0]
    print(summary_price)
    summary_pct = stock_data.loc[stock_data['Date'].isin(
        [stock_data['Date'].min(), summary_date]), 'Close'].pct_change()#.values[1]
    print(summary_pct)
    summary_pct = summary_pct.values[1]
    
    
    # Stock Price Chart
    stock_fig = go.Figure()
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"],
                                   mode='lines', name='Closing Price', line=dict(color='blue' if switch_on else 'coral')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["50_MA"],
                                   mode='lines', name='50-Day MA', line=dict(color='red', dash='dot')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["200_MA"],
                                   mode='lines', name='200-Day MA', line=dict(color='orange', dash='dot')))
    stock_fig.add_trace(go.Scatter(x=future_dates, y=future_prices,
                                   mode='lines', name='Predicted Prices', line=dict(color='green' if switch_on else 'magenta', dash='dot')))
    stock_fig.update_layout(
        title=f"{selected_stock} Stock Price & Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
    # fig_df = pd.concat([stock_data[["Date","Close","50_MA","200_MA",]],pd.DataFrame(future_dates,future_prices,columns=['Date','Predicted_Price'])],sort=False)
    stock_data['Date'] = stock_data['Date'].dt.normalize().dt.date
    # update_theme(is_dark) #Does not work
    stock_fig.layout.template = get_template(switch_on)
    return stock_data.sort_values(by='Date', ascending=False).to_dict('records'), stock_fig, create_stock_card(summary_date.date(),summary_price, summary_pct), None
    # print(type(future_dates),type(future_prices))
    # print(future_dates,future_prices)
    # pdf = pd.DataFrame(zip(future_dates,future_prices),columns=['Date','Predicted Price'])
    # pdf['Date'] = pdf['Date'].dt.date
    # pdf['Predicted Price'] = pdf['Predicted Price'].round(2)
    # return pdf.to_dict('records'), stock_fig, None#fig_df,
    # return stock_data[['Date', 'Close']].tail().to_dict('records'), stock_fig, None#fig_df,


@callback(
    Output('download', "data"),
    Input('download-button', "n_clicks"),
    State('stock-selector', 'value'),
    State('input-box', 'value'),
    State("file-type-selector", 'value'),
    State('data-table', "derived_virtual_data"),
    prevent_initial_call=True,
    # prevent_initial_callbacks=True,
)
def download_data(click, stock_selected, input_value, download_type, data):
    '''
    Download data as CSV or XLSX from DataTable
    '''
    if not click:
        return no_update
    stock = stock_selected if not input_value else input_value
    dff = pd.DataFrame(data)
    dff.set_index(dff.columns[0], inplace=True)
    if download_type == "csv":
        writer = dff.to_csv
    else:
        writer = dff.to_excel
    return dcc.send_data_frame(writer, f"{stock} Price Data.{download_type}")


def get_template(switch_on):
    '''
    Get light or dark template
    '''
    return pio.templates[template_themes[0]] if switch_on else pio.templates[template_themes[1]]


@callback(
    Output("stock-price-chart", "figure", allow_duplicate=True),
    Output('data-table', 'style_header'),
    Output('data-table', 'style_data_conditional'),
    Output('data-table', 'style_data'),
    Input("color-mode-switch", "value"),
    prevent_initial_call=True
)
def update_theme(switch_on):
    '''
    Update theme to light or dark
    '''
    # switch_on = Light
    # When using Patch() to update the figure template, you must use the figure template dict
    # from plotly.io  and not just the template name
    # template_themes list is ordered light to dark
    # pio.templates[template_themes[0]] if switch_on else pio.templates[template_themes[1]]
    template = get_template(switch_on)

    patched_figure = Patch()
    patched_figure["layout"]["template"] = template
    styles = data_table_style(not (switch_on))
    return patched_figure, styles['style_header'], styles['style_data_conditional'], styles['style_data']


clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)

