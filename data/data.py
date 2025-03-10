import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from pmdarima import auto_arima
from datetime import timedelta, datetime
import os
from .db import initial_setup, get_ticker_id, get_ticker_prices, insert_stock_price_data, upsert_stock_price_data, ticker_price_last_update


initial_setup()
# Cache model predictions
predictions_cache = {}



def get_new_data(ticker,**kwargs):
    '''
    Download stock data from yahoo finance.
    '''
    stock_data = yf.download(ticker, period=kwargs.pop('period',"5y"), auto_adjust=kwargs.pop('auto_adjust',True),multi_level_index=kwargs.pop('multi_level_index',False),**kwargs)
    if stock_data is None or stock_data.empty:
        return None
    stock_data.reset_index(inplace=True)
    return stock_data

# Fetch Stock Data


def fetch_stock_data(ticker,deprecated=False):
    '''
    Download stock data from yahoo finance.
    Adding additional features for price prediction.
    '''
    
    
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

    if stock_data is None or stock_data.empty or needs_update:
        stock_data = get_new_data(ticker)
        # stock_data.columns = stock_data.columns.droplevel(1)
        # stock_data.reset_index(inplace=True)
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

    if stock_data is None or stock_data.empty:
        return None
    print(stock_data.head())
    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    return add_features(stock_data)

def add_features(stock_data):
    '''
    Add features (metrics) for the predictive model
    '''
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
