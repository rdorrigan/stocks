from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String,Float, DateTime, text
import pandas as pd
import os

# Initialize SQLAlchemy engine


def get_engine():
    '''
    Get sqlalchemy engine
    '''
    return create_engine('sqlite:///data.db', echo=True)

def db_exists():
    return os.path.exists('data.db')

def create_db(df,override=False):
    '''
    Create database with tickers table
    '''
    if not db_exists() or override:
        # SQLAlchemy engine & metadata
        engine = get_engine()
        metadata = MetaData()

        # Define symbols table (equivalent to a pandas DataFrame)
        _ = Table('tickers', metadata,
                              Column('TickerID', Integer, primary_key=True),
                              Column('Ticker', String),
                              Column('Label', String),
                              Column('MarketCap', Float),

                              )

        # Create the table if it doesn't exist
        metadata.create_all(engine)

        # Insert DataFrame into SQLAlchemy table
        with engine.connect() as conn:
            df.to_sql('tickers', conn, if_exists='replace', index=False)

def list_tables():
    '''
    List table names
    '''
    engine = get_engine()
    with engine.connect() as connection:
        result = connection.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = result.fetchall()
        return [table[0] for table in tables]

def has_table(table):
    '''
    tickers table exists
    '''
    return table in list_tables()

def get_all_tickers(order_by_col=None,ascending=False):
    engine = get_engine()
    # Fetch data using SQLAlchemy
    with engine.connect() as conn:
        query = "SELECT * FROM tickers"
        if order_by_col:
            query += f'\norder by {order_by_col} {"desc" if not ascending else ""}'
        df = pd.read_sql(query, conn)
    return df

def get_top_tickers():
    '''
    Sort tickers by MarketCap
    '''
    return get_all_tickers('MarketCap')


def get_top_stocks():
    import yfinance as yf
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "GOOGL",
                       "META", "BRK-B", "UNH", "XOM", "AMD", "APLD", "SOUN", "INTC",
                       "TSM"]
    data = []
    for ticker in tickers:
        tick = yf.Ticker(ticker)
        short_name = tick.info['shortName']
        market_cap = tick.fast_info.market_cap
        data.append({"Ticker": ticker, "Label": f"{short_name} ({ticker})","MarketCap" : market_cap})
    df = pd.DataFrame(data).sort_values(by=['Ticker'])
    
    # stock_info = {ticker: yf.Ticker(ticker).info for ticker in tickers}
    # sorted_stocks = sorted(stock_info.items(), key=lambda x: x[1].get(
    #     "marketCap", 0), reverse=True)
    # df = pd.DataFrame([{"Ticker": ticker, "Label": f"{info['shortName']} ({ticker})"}
    #                   for ticker, info in sorted_stocks])
    df.reset_index(names='TickerID', inplace=True)
    df['TickerID'] = df['TickerID'] + 1
    print(df.head())
    return df


def initial_setup(override=False):
    if db_exists():
        if not has_table('tickers'):
            df = get_top_stocks()
            # Call to create database and table on the first run
            create_db(df,override=override)
        if not has_table('stock_prices'):
            create_stock_price_table()
    else:
        df = get_top_stocks()
        # Call to create database and table on the first run
        create_db(df,override=True)
        create_stock_price_table()
    # # Example of loading data
    # df_from_db = load_data_from_db()
    # print(df_from_db)



def create_stock_price_table():
    engine = get_engine()
    metadata = MetaData()

    _ = Table('stock_prices', metadata,
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
 
def query_db(query):
    '''
    Read query
    '''
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
        df['Date'] = df['Date'].dt.date
        conn.execute(
            text("""
            INSERT OR REPLACE INTO stock_prices 
            (Ticker, Date, Open, High, Low, Close, Volume)
            VALUES (:Ticker, :Date, :Open, :High, :Low, :Close, :Volume)
            """), df.to_dict(orient='records')
            # [
            #     {"ticker": t[0], "date": t[1], "open": t[2], "high": t[3], "low": t[4], "close": t[5], "volume": t[6]}
            #     for t in df.itertuples(index=False, name=None)
            # ]
        )
def replace_stock_price_data(df):
    engine = get_engine()
    df.to_sql('stock_prices', engine, if_exists='replace', index=False)

def get_ticker_id(ticker):
    query = f"SELECT ID FROM stock_prices WHERE Ticker = '{ticker}' LIMIT 1"
    try:    
        return query_db(query)
    except Exception as e:
        print(f"Database error: {e}")
        return None

def ticker_price_last_update(ticker):
    query = f"""
    SELECT MAX(DATE) AS DATE 
    FROM STOCK_PRICES 
    WHERE TICKER = '{ticker}'"""
    return query_db(query)
    

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
        return query_db(query)
    except Exception as e:
        print(f"Database error: {e}")
        return None