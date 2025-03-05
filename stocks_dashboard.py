import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime
import os
import glob
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

# Initialize the app
app = dash.Dash(__name__)
server = app.server  # Needed for deployment

# Ensure a valid port is assigned
PORT = int(os.environ.get("PORT", 8050))

# Layout
app.layout = html.Div([
    html.H1("Stock Analysis Dashboard", style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id="stock-selector",
        options=[
            {"label": "Apple (AAPL)", "value": "AAPL"},
            {"label": "Microsoft (MSFT)", "value": "MSFT"},
            {"label": "Tesla (TSLA)", "value": "TSLA"},
            {"label": "Amazon (AMZN)", "value": "AMZN"},
            {"label": "Netflix (NFLX)","value" : "NFLX"},
            {"label": "Nvidia (NVDA)","value" : "NVDA"},
            {"label": "Advanced Micro Devices Inc (AMD)","value" : "AMD"},
            {"label": "Applied Digital Corp (APLD)","value" : "APLD"},
            {"label": "SoundHound AI Inc (SOUN)","value" : "SOUN"}
        ],
        value="AAPL",
        clearable=False,
        style={'width': '50%', 'margin': 'auto'}
    ),
    
    dcc.Graph(id="stock-price-chart"),
    dcc.Graph(id="prediction-chart"),
])
SAVE_DIR = os.path.join(os.environ['USERPROFILE'],'Documents/Python Scripts/Stocks/')
def stock_data_file_formatter(ticker):
    dt = datetime.now().isoformat(timespec='seconds').replace(':','_')
    return os.path.join(SAVE_DIR,f'{ticker} stock_data {dt}.csv')
def get_latest_stock_data_file(ticker):
    files = glob.glob(f'{ticker}*')
    if files:
        return max(files,os.path.getmtime)
    return None
def get_file_timestamp(latest_file):
    return datetime.fromisoformat(latest_file.split('stock data ')[1][:-4].replace('_',':'))
# Fetch Stock Data
def fetch_stock_data(ticker):
    def get_new_data(ticker):
        return yf.download(ticker, period="3y")
    def save_new_data(stock_data):
        stock_data.to_csv(stock_data_file_formatter(ticker))
    def read_stock_data(file):
        return pd.read_csv(file,index_col=[0,1])
    now = datetime.now()
    latest_file = get_latest_stock_data_file(ticker)
    if latest_file:
        file_timestamp = get_file_timestamp(latest_file)
        if (now - file_timestamp).total_seconds > 60 ** 2:
            stock_data = get_new_data(ticker)
            save_new_data(stock_data)
        else:
            stock_data = read_stock_data(latest_file)            
    else:
        stock_data = get_new_data(ticker)
        save_new_data(stock_data)
    

    stock_data.columns = stock_data.columns.droplevel(1)
    stock_data.reset_index(inplace=True)
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data.sort_values("Date", inplace=True)
    
    # Calculate Moving Averages
    stock_data["50_MA"] = stock_data["Close"].rolling(window=50, min_periods=1).mean()
    stock_data["200_MA"] = stock_data["Close"].rolling(window=200, min_periods=1).mean()
    # print(stock_data.dtypes)
    return stock_data
    

# Train Prediction Model
def train_prediction_model(stock_data):
    if stock_data is None or stock_data.empty:
        return [], []
    # print(stock_data.columns.values.tolist())
    stock_data = stock_data.dropna(subset=["Close"])
    stock_data = stock_data[stock_data["Date"].notna()]
    
    X = stock_data["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = stock_data["Close"].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_dates = [stock_data["Date"].max() + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_prices = model.predict(future_X)
    
    return future_dates, future_prices

def train_arima_model(stock_data):
    from pmdarima import auto_arima
    if stock_data is None or stock_data.empty:
        return [], []
    # print(stock_data.columns.values.tolist())
    stock_data = stock_data.dropna(subset=["Close"])
    stock_data = stock_data[stock_data["Date"].notna()]
    
    X = stock_data["Date"].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y = stock_data["Close"].values
    model_arima = auto_arima(y, seasonal=False, trace=True, stepwise=True)

    # Predict future values
    forecast_steps = 30
    future_dates = [stock_data["Date"].max() + timedelta(days=i) for i in range(1, forecast_steps + 1)]
    
    future_forecast = model_arima.predict(n_periods=forecast_steps)#, return_conf_int=True
    # print(conf_int)
    return future_dates,future_forecast

# Callback to update graphs
@app.callback(
    [Output("stock-price-chart", "figure"),
     Output("prediction-chart", "figure")],
    [Input("stock-selector", "value")]
)
def update_graphs(selected_stock):
    stock_data = fetch_stock_data(selected_stock)
    if stock_data is None or stock_data.empty:
        return go.Figure(), go.Figure()
    try:
        future_dates, future_prices = train_arima_model(stock_data)
    except ValueError:
        future_dates, future_prices = train_prediction_model(stock_data)

    # Stock Price Chart
    stock_fig = go.Figure()
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"],
                                   mode='lines', name='Closing Price', line=dict(color='blue')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["50_MA"],
                                   mode='lines', name='50-Day MA', line=dict(color='red', dash='dot')))
    stock_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["200_MA"],
                                   mode='lines', name='200-Day MA', line=dict(color='green', dash='dot')))
    stock_fig.update_layout(title=f"{selected_stock} Stock Price & Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
    
    # Prediction Chart
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=stock_data["Date"], y=stock_data["Close"],
                                  mode='lines', name='Historical Prices', line=dict(color='blue')))
    if future_dates:
        pred_fig.add_trace(go.Scatter(x=future_dates, y=future_prices,
                                      mode='lines', name='Predicted Prices', line=dict(color='red', dash='dot')))
    pred_fig.update_layout(title=f"{selected_stock} 30-Day Price Prediction", xaxis_title="Date", yaxis_title="Predicted Price (USD)")
    
    return stock_fig, pred_fig

# Run the app
if __name__ == "__main__":
    try:
        app.run_server(debug=True) #, host="0.0.0.0", port=PORT
    except Exception as e:
        print(f"Error starting server: {e}")
