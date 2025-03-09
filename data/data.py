import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pmdarima import auto_arima
from datetime import timedelta, datetime
import os
import glob
from .db import initial_setup, get_ticker_id, get_ticker_prices, insert_stock_price_data, upsert_stock_price_data, ticker_price_last_update

SAVE_DIR = os.path.join(os.getenv('USERPROFILE', ''),
                            'Documents/Python Scripts/Stocks/')
initial_setup()
# Cache model predictions
predictions_cache = {}

def get_top_stocks():
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL", "NFLX",
                       "META", "BRK-B", "UNH", "XOM", "AMD", "APLD", "SOUN", "INTC",
                       "TSM"]
    stock_info = {ticker: yf.Ticker(ticker).info for ticker in tickers}
    sorted_stocks = sorted(stock_info.items(), key=lambda x: x[1].get(
        "marketCap", 0), reverse=True)
    return [{"label": f"{info['shortName']} ({ticker})", "value": ticker} for ticker, info in sorted_stocks]


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


def fetch_stock_data(ticker,deprecated=False):
    '''
    Download stock data from yahoo finance.
    Adding additional features for price prediction.
    '''
    def get_new_data(ticker):
        '''
        Download stock data from yahoo finance.
        '''
        stock_data = yf.download(ticker, period="5y", auto_adjust=True)
        if stock_data is None or stock_data.empty:
            return None
        if deprecated:
            # if not PROD:
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
    if not deprecated:
        # if PROD:
        needs_update = False
        stock_data = None
        ticker_id = get_ticker_id(ticker)
        print(ticker_id)
        if not ticker_id.empty:
            last_update = datetime.fromisoformat(
                ticker_price_last_update(ticker).iloc[0, 0])
            print(last_update)
            if last_update.date() < datetime.today().date():
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
