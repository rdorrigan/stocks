from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String,Float, DateTime, text
import pandas as pd
import os

# Initialize SQLAlchemy engine


def get_engine():
    return create_engine('sqlite:///data.db', echo=True)

# Function to create the DB and table if not already created


def create_db(df,override=False):
    if not os.path.exists('data.db') or override:
        # SQLAlchemy engine & metadata
        engine = get_engine()
        metadata = MetaData()

        # Define symbols table (equivalent to a pandas DataFrame)
        tickers_table = Table('tickers', metadata,
                              Column('TickerID', Integer, primary_key=True),
                              Column('Ticker', String),
                              Column('Label', String),

                              )

        # Create the table if it doesn't exist
        metadata.create_all(engine)

        # Insert DataFrame into SQLAlchemy table
        with engine.connect() as conn:
            df.to_sql('tickers', conn, if_exists='replace', index=False)

def list_tables():
    engine = get_engine()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = result.fetchall()
        return [table[0] for table in tables]

def load_data_from_db():
    engine = get_engine()
    # Fetch data using SQLAlchemy
    with engine.connect() as conn:
        query = "SELECT * FROM tickers"
        df = pd.read_sql(query, conn)
    return df


def get_top_stocks():
    import yfinance as yf
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL",
                       "META", "BRK-B", "UNH", "XOM", "AMD", "APLD", "SOUN", "INTC",
                       "TSM"]
    stock_info = {ticker: yf.Ticker(ticker).info for ticker in tickers}
    sorted_stocks = sorted(stock_info.items(), key=lambda x: x[1].get(
        "marketCap", 0), reverse=True)
    df = pd.DataFrame([{"Ticker": ticker, "Label": f"{info['shortName']} ({ticker})"}
                      for ticker, info in sorted_stocks])
    df.reset_index(names='TickerID', inplace=True)
    df['TickerID'] = df['TickerID'] + 1
    print(df.head())
    return df


def initial_setup():
    df = get_top_stocks()
    # Call to create database and table on the first run
    create_db(df,override=True)
    # Example of loading data
    df_from_db = load_data_from_db()
    print(df_from_db)



def create_stock_price_table():
    engine = get_engine()
    metadata = MetaData()

    stock_price_table = Table('stock_prices', metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('Ticker', String, nullable=False),
        Column('Date', DateTime, nullable=False),
        Column('Open', Float),
        Column('High', Float),
        Column('Low', Float),
        Column('Close', Float),
        Column('Volume', Integer),
    )

    # Create table with a UNIQUE constraint on (Ticker, Date)
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Ticker TEXT NOT NULL,
            Date DATETIME NOT NULL,
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INTEGER,
            UNIQUE(Ticker, Date)
        )
        """))
 
def query_stock_prices(query):
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(query, conn)
        

def insert_stock_price_data(df):
    engine = get_engine()
    with engine.connect() as conn:
        df.to_sql('stock_prices', conn, if_exists='append', index=False)
def upsert_stock_price_data(df):
    engine = get_engine()
    with engine.connect() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT OR REPLACE INTO stock_prices (Ticker, Date, Open, High, Low, Close, Volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (row['Ticker'], row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])))
def replace_stock_price_data(df):
    engine = get_engine()
    df.to_sql('stock_prices', engine, if_exists='replace', index=False)

def get_ticker_id(ticker):
    query = f"SELECT ID FROM stock_prices WHERE Ticker = '{ticker}' LIMIT 1"
    try:
        return query_stock_prices(query)
    except Exception as e:
        print(f"Database error: {e}")
        return None

def ticker_price_last_update(ticker):
    query = f"""
    SELECT MAX(DATE) AS DATE 
    FROM STOCK_PRICES 
    WHERE TICKER = '{ticker}'"""
    return query_stock_prices(query)
    

def get_ticker_prices(ticker):
    '''
    Get all prices for ticker(s)
    '''
    if isinstance(ticker,str):
        query = f"SELECT * FROM stock_prices WHERE Ticker = '{ticker}'"
    else:
        tickers = '('
        for t in ticker:
            tickers += f"'{t}',"
        tickers += ')'
        query = f"SELECT * FROM stock_prices WHERE Ticker IN {tickers}"
    try:
        return query_stock_prices(query)
    except Exception as e:
        print(f"Database error: {e}")
        return None