from dash import Dash, dcc, html, Input, Output, State, Patch, no_update, clientside_callback
from dash.dash_table import DataTable, FormatTemplate
from dash_bootstrap_templates import load_figure_template
import plotly.io as pio
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pmdarima import auto_arima
from datetime import timedelta, datetime
import os
import glob
import argparse
import warnings
from db.db import initial_setup,get_ticker_id,get_ticker_prices,insert_stock_price_data,upsert_stock_price_data,ticker_price_last_update
warnings.filterwarnings('ignore', category=FutureWarning)
PROD = True
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.2.4/dbc.min.css"
# adds  templates to plotly.io
# Vizro is the best ['bootstrap','bootstrap_dark'], ["minty", "minty_dark"]
template_themes = ['vizro', 'vizro_dark']
load_figure_template(template_themes)
# Initialize the app
dash_app = Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP, dbc.themes.MINTY, dbc.icons.FONT_AWESOME, dbc_css])
# server = dash_app.server  # Needed for deployment


color_mode_switch = html.Span(
    [
        dbc.Label(className="fa fa-moon", html_for="color-mode-switch"),
        dbc.Switch(id="color-mode-switch", value=False,
                   className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="color-mode-switch"),
    ]
)
# Cache model predictions
predictions_cache = {}

# Ensure a valid port is assigned
PORT = int(os.environ.get("PORT", 8050))

# Fetch top stocks dynamically

initial_setup()

def get_top_stocks():
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL",
                       "META", "BRK-B", "UNH", "XOM", "AMD", "APLD", "SOUN", "INTC",
                       "TSM"]
    stock_info = {ticker: yf.Ticker(ticker).info for ticker in tickers}
    sorted_stocks = sorted(stock_info.items(), key=lambda x: x[1].get(
        "marketCap", 0), reverse=True)
    return [{"label": f"{info['shortName']} ({ticker})", "value": ticker} for ticker, info in sorted_stocks]


# Components
children = []

header = html.H1("Stock Price Dashboard", style={'textAlign': 'center'})
drop_down_header = html.H4("Stock Selector", style={'textAlign': 'center'})
drop_down_input = html.Div(children=[
    dcc.Dropdown(
        id="stock-selector",
        options=get_top_stocks(),
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
        # , 'margin': 'auto','textAlign': 'center'
        style={'width': '20%', 'marginTop': '10px'},
    ),
    dcc.Store(id="error-store"),  # Hidden storage for errors
    dbc.Alert(id="error-alert", color="danger",
              is_open=False, dismissable=True)  # Alert box
], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'},
    className='dbc'
)

price_graph = dcc.Graph(id="stock-price-chart")
model_header = html.H4("Predictive Model Selector",
                       style={'textAlign': 'center', 'margin': '10px'})
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
period_labels = ['5D', '1M', '3M', '6M', '1Y', '2Y', '3Y']
date_period_options = []
for l in period_labels:
    period_options = {}
    period_options["label"] = l
    period_options["value"] = l
    date_period_options.append(period_options)
date_period_selector = dcc.RadioItems(
    id="date-period-selector",
    options=date_period_options,
    value="3Y",
    labelStyle={'display': 'inline-block', 'margin': '10px'}
)
date_header = html.H4("Date Filters",
                      style={'textAlign': 'center', 'margin': '10px'})

model_div = html.Div(children=[model_header, model_selector], style={
                     'display': 'flex', 'flexDirection': 'column', 'alignItems': 'left'})
date_div = html.Div(children=[date_header, date_period_selector], style={
                    'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
model_date_div = date_div = html.Div(children=[model_div, date_div], style={
                                     'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'})
# prediction_graph = dcc.Graph(id="prediction-chart")
money = FormatTemplate.money(2)
percent = FormatTemplate.percentage(2, True)


def data_table_style(dark) -> dict:
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


data_table = DataTable(id='data-table', page_size=10, columns=[
    {'name': 'Date', 'id': 'Date', 'type': 'datetime'},
    {'name': 'Price', 'id': 'Close', 'format': money, "type": 'numeric'},
    {'name': '50 Day MA', 'id': '50_MA', 'format': money, "type": 'numeric'},
    {'name': '200 Day MA', 'id': '200_MA', 'format': money, "type": 'numeric'},
    {'name': 'Daily Return', 'id': 'Daily_Return',
        'format': percent, "type": 'numeric'},
    {'name': 'Volatility', 'id': 'Volatility', 'format': percent, "type": 'numeric'}],
    sort_action="native",
    filter_action='native',
    style_table={"overflowX": "auto"},
    **data_table_style(False))
file_type_header = html.H4("Choose download file type.",
                           style={'textAlign': 'left'})
file_type_selector = dcc.RadioItems(id="file-type-selector", options=[{"label": "Excel file", "value": "xlsx"},
                                                                      {"label": "CSV file",
                                                                          "value": "csv"}
                                                                      ], value='csv',
                                    labelStyle={
                                        'display': 'inline-block', 'margin': '10px'}
                                    )
download_button = html.Button(
    "Download Data", id='download_prices', style={"marginTop": 20})
download_component = dcc.Download(id='Download')
file_div = html.Div(children=[file_type_header, file_type_selector],
                    id='file-div', style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'},)
download_div = html.Div(children=[download_button, download_component],
                        id='download-div', style={'display': 'flex', 'flex-direction': 'column', 'margin': '10px'},)
file_download_div = html.Div(children=[file_div, download_div],
                             id='file-download-div', style={'display': 'flex', 'flex-direction': 'row', 'margin': '10px'},)
# List of children components that will display in order
children.extend([header, color_mode_switch, drop_down_header, drop_down_input,
                 model_date_div,
                #  model_header, model_selector,
                 price_graph, file_download_div,
                 data_table])


# Layout
dash_app.layout = html.Div(children=children, id='container', className="dbc")
if not PROD:
    SAVE_DIR = os.path.join(os.getenv('USERPROFILE', ''),
                            'Documents/Python Scripts/Stocks/')
else:
    SAVE_DIR = ''


def stock_data_file_formatter(ticker):
    '''
    Create file name
    '''
    dt = datetime.now().isoformat(timespec='seconds').replace(':', '_')
    return os.path.join(SAVE_DIR, f'{ticker} stock_data {dt}.csv')


def get_latest_stock_data_file(ticker):
    '''
    Get the latest CSV file and remove any excess for the given ticker
    '''
    files = glob.glob(f'{SAVE_DIR}/{ticker}*')
    if files:
        latest_file = max(files, key=os.path.getmtime)
        for f in files:
            if f != latest_file:
                os.remove(f)
        return latest_file
    return None


def get_file_timestamp(latest_file):
    '''
    Parse timestamp
    '''
    return datetime.fromisoformat(latest_file.split('stock_data ')[1][:-4].replace('_', ':'))
# Fetch Stock Data


def fetch_stock_data(ticker):
    '''
    Download stock data from yahoo finance.
    Adding additional features for price prediction.
    '''
    def get_new_data(ticker):
        '''
        Download stock data from yahoo finance.
        '''
        stock_data = yf.download(ticker, period="3y", auto_adjust=True)
        if stock_data is None or stock_data.empty:
            return None
        if not PROD:
            save_new_data(stock_data)
        return stock_data

    def save_new_data(stock_data):
        '''
        Save stock data
        '''
        stock_data.to_csv(stock_data_file_formatter(ticker))

    def read_stock_data(file):
        '''
        Read a saved csv
        '''
        return pd.read_csv(file, header=[0, 1], index_col=0)
    if PROD:
        needs_update =  False
        stock_data = None
        ticker_id = get_ticker_id(ticker)
        print(ticker_id)
        if not ticker_id.empty:
            last_update = datetime.fromisoformat(ticker_price_last_update(ticker).iloc[0,0])
            print(last_update)
            if last_update.date() <  datetime.today().date():
                needs_update = True
            else:
                # Query database for existing data
                stock_data = get_ticker_prices(ticker)

        if stock_data is None or stock_data.empty:
            stock_data = get_new_data(ticker)
            stock_data.columns = stock_data.columns.droplevel(1)
            stock_data.reset_index(inplace=True)
            sdf = stock_data.copy()
            
            
            cols = sdf.columns.values.tolist()
            sdf['Ticker'] = ticker
            reorder = ['Ticker']
            reorder.extend(cols)
            sdf = sdf[reorder]
            if needs_update:
                upsert_stock_price_data(sdf)
            else:
                insert_stock_price_data(sdf)
            del sdf

    else:
        now = datetime.now()
        latest_file = get_latest_stock_data_file(ticker)
        if latest_file:
            file_timestamp = get_file_timestamp(latest_file)
            if (now - file_timestamp).total_seconds() > 60 ** 2:
                stock_data = get_new_data(ticker)   

            else:
                stock_data = read_stock_data(latest_file)
        else:
            stock_data = get_new_data(ticker)
        stock_data.columns = stock_data.columns.droplevel(1)
        stock_data.reset_index(inplace=True)

    
    if stock_data is None or stock_data.empty:
        return None
    print(stock_data.head())
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data.sort_values("Date", inplace=True)
    stock_data = stock_data[['Date', 'Close']]
    # Calculate Moving Averages
    stock_data["50_MA"] = stock_data["Close"].rolling(
        window=50, min_periods=1).mean()
    stock_data["200_MA"] = stock_data["Close"].rolling(
        window=200, min_periods=1).mean()
    # Additional Features
    stock_data["Daily_Return"] = stock_data["Close"].pct_change().fillna(0)
    stock_data["Volatility"] = stock_data["Daily_Return"].rolling(
        window=10, min_periods=1).std()
    stock_data["Close"] = stock_data["Close"].round(2)
    # print(stock_data.dtypes)
    return stock_data


def screen(key):
    '''
    https://yfinance-python.org/reference/api/yfinance.screen.html
    '''
    # Predefined queries
    # ['day_gainers','day_losers']
    assert key in yf.PREDEFINED_SCREENER_QUERIES

    def create_label(quote):
        return f"{quote['displayName']} ({quote['symbol']})"
    result = yf.screen(key)
    if 'quotes' in result:
        quotes = result['quotes']
        del result
        for quote in quotes:
            quote['label'] = create_label(quote)
        return quotes
    return None
# Train Prediction Model


def train_prediction_model(stock_data, model_type='XGB'):
    '''
    Predict future prices useing additional features for a few model types
    '''
    if stock_data is None or stock_data.empty:
        return [], []
    stock_data = stock_data.sort_values("Date")
    stock_data["Date_Ordinal"] = stock_data["Date"].map(pd.Timestamp.toordinal)

    feature_columns = ["Date_Ordinal", "50_MA", "200_MA", "Volatility"]
    stock_data = stock_data.dropna()
    X = stock_data[feature_columns]
    y = stock_data["Close"].values
    forecast_steps = 30
    future_dates = [stock_data["Date"].max() + timedelta(days=i)
                    for i in range(1, forecast_steps + 1)]
    future_dates_ordinal = np.array(
        [d.toordinal() for d in future_dates]).reshape(-1, 1)

    if model_type == 'lightgbm':
        model = LGBMRegressor(n_estimators=100, verbose=-1)
    elif model_type == 'xgboost':
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    elif model_type == 'arima':
        # Extend features for predictions dynamically
        recent_data = stock_data.iloc[-50:]
        future_moving_avg_50 = recent_data["Close"].rolling(
            window=50, min_periods=1).mean().values[-1]
        future_moving_avg_200 = recent_data["Close"].rolling(
            window=200, min_periods=1).mean().values[-1]
        future_volatility = recent_data["Daily_Return"].rolling(
            window=10, min_periods=1).std().values[-1]

        future_X = np.column_stack((future_dates_ordinal, np.full((forecast_steps,), future_moving_avg_50),
                                    np.full((forecast_steps,), future_moving_avg_200), np.full((forecast_steps,), future_volatility)))

        model_arima = auto_arima(y, X, seasonal=False,
                                 trace=True, stepwise=True)
        future_prices = model_arima.predict(
            X=future_X, n_periods=forecast_steps)
        return future_dates, future_prices
    else:
        model = LinearRegression()

    model.fit(X, y)
    # Extend features dynamically
    future_X = []
    recent_data = stock_data.iloc[-50:].copy()
    for i in range(forecast_steps):
        future_moving_avg_50 = recent_data["Close"].rolling(
            window=50, min_periods=1).mean().values[-1]
        future_moving_avg_200 = recent_data["Close"].rolling(
            window=200, min_periods=1).mean().values[-1]
        future_volatility = recent_data["Daily_Return"].rolling(
            window=10, min_periods=1).std().values[-1]
        future_X.append([future_dates_ordinal[i, 0], future_moving_avg_50,
                         future_moving_avg_200, future_volatility])

        # Simulate closing price for next day
        next_close = model.predict(np.array([future_X[-1]]))[0]
        next_return = (
            next_close - recent_data["Close"].values[-1]) / recent_data["Close"].values[-1]
        new_row = pd.DataFrame({
            "Date": [future_dates[i]],
            "Close": [next_close],
            "Daily_Return": [next_return]
        })
        recent_data = pd.concat([recent_data, new_row], ignore_index=True)

    future_X = np.array(future_X)

    # future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_prices = model.predict(future_X)
    # print(dict(zip(future_dates,future_prices)))
    return future_dates, future_prices


def get_predictions(selected_stock, stock_data, model_type):
    '''
    Get cached prediction prices and or save new ones to cache
    '''
    stock_key = f"{selected_stock}-{model_type}"
    if stock_key not in predictions_cache:
        future_dates, future_prices = train_prediction_model(
            stock_data, model_type)
        predictions_cache[stock_key] = (future_dates, future_prices)
    return predictions_cache[stock_key]

# UGGG
# @dash_app.callback(
# 	[Output('stock-price-chart', 'style'),
# 	 Output('prediction-chart', 'style'),
# 	 Output('stock-price-chart', 'delay_show'),
#    Output('prediction-chart', 'delay_show'),],
# 	[Input('stock-price-chart', 'is_loading'),
# 	 Input('prediction-chart', 'is_loading')]
# )
# def update_opacity(stock_price_loading_state,prediction_loading_state):
# 	# Set opacity to 0.5 when updating
# 	update = []
# 	for is_loading in [stock_price_loading_state,prediction_loading_state]:
# 		if is_loading:
# 			update.append({'opacity': 0.5})  # Make the graph semi-transparent during update
# 		else:
# 			update.append({'opacity': 1})
# 	return update

# Display error message


@dash_app.callback(
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
@dash_app.callback(Output('input-box', 'value'), Input("stock-selector", "value")
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


@dash_app.callback(
    [
        Output('data-table', "data"),
        Output("stock-price-chart", "figure", allow_duplicate=True),
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
        return [{c: '' for c in stock_data_cols}], go.Figure(), f"Invalid stock symbol: {input_value}"

    # Train model or get cached model predictions
    future_dates, future_prices = get_predictions(
        selected_stock, stock_data, model_type)
    stock_data = stock_data[stock_data_cols]
    stock_data[stock_data_cols[2:]] = stock_data[stock_data_cols[2:]].round(2)
    if date_filter not in ('3Y', None):
        date_range = period_to_date_range(date_filter)
        tdf = stock_data.loc[stock_data['Date'].between(
            *date_range, inclusive='both'), :]
        if not tdf.empty:
            stock_data = tdf.copy()
            del tdf
    # Stock Price Chart
    stock_fig = go.Figure()
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"],
                                   mode='lines', name='Closing Price', line=dict(color='blue')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["50_MA"],
                                   mode='lines', name='50-Day MA', line=dict(color='red', dash='dot')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["200_MA"],
                                   mode='lines', name='200-Day MA', line=dict(color='orange', dash='dot')))
    stock_fig.add_trace(go.Scatter(x=future_dates, y=future_prices,
                                   mode='lines', name='Predicted Prices', line=dict(color='green', dash='dot')))
    stock_fig.update_layout(
        title=f"{selected_stock} Stock Price & Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
    # fig_df = pd.concat([stock_data[["Date","Close","50_MA","200_MA",]],pd.DataFrame(future_dates,future_prices,columns=['Date','Predicted_Price'])],sort=False)
    stock_data['Date'] = stock_data['Date'].dt.date
    # update_theme(is_dark) #Does not work
    stock_fig.layout.template = get_template(switch_on)
    return stock_data.sort_values(by='Date', ascending=False).to_dict('records'), stock_fig, None
    # print(type(future_dates),type(future_prices))
    # print(future_dates,future_prices)
    # pdf = pd.DataFrame(zip(future_dates,future_prices),columns=['Date','Predicted Price'])
    # pdf['Date'] = pdf['Date'].dt.date
    # pdf['Predicted Price'] = pdf['Predicted Price'].round(2)
    # return pdf.to_dict('records'), stock_fig, None#fig_df,
    # return stock_data[['Date', 'Close']].tail().to_dict('records'), stock_fig, None#fig_df,


@dash_app.callback(
    Output(download_component, "data"),
    Input(download_button, "n_clicks"),
    State('stock-selector', 'value'),
    State('input-box', 'value'),
    State(file_type_selector, 'value'),
    State(data_table, "derived_virtual_data"),
    prevent_initial_call=True,
    # prevent_initial_callbacks=True,
)
def download_data(click, stock_selected, input_value, download_type, data):
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
    return pio.templates[template_themes[0]] if switch_on else pio.templates[template_themes[1]]


@dash_app.callback(
    Output("stock-price-chart", "figure", allow_duplicate=True),
    Output(data_table, 'style_header'),
    Output(data_table, 'style_data_conditional'),
    Output(data_table, 'style_data'),
    Input("color-mode-switch", "value"),
    prevent_initial_call=True
)
def update_theme(switch_on):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prod', action='store_true')
    return parser.parse_args()


# Run the apps
if __name__ == "__main__":
    args = parse_args()
    if args.prod:
        PROD = True
    try:
        if not PROD:
            dash_app.run(debug=True)  # , host="0.0.0.0", port=PORT
        else:
            from waitress import serve
            serve(dash_app.server, host="0.0.0.0", port=PORT)
    except Exception as e:
        print(f"Error starting server: {e}")