import pandas as pd
from utils import calculate_indicators
import yfinance as yf
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists,create_database

def db_create(engine_url, dataframe):
    engine = create_engine(engine_url)

    if not database_exists(engine.url):
        create_database(engine.url)

    if database_exists(engine.url):
        data_type = 'data'
        print('Populating database with', data_type)
        dataframe.to_sql(data_type, engine)

def download_data(symbols):
    return yf.download(
            tickers = symbols,
            period = 'max',
            interval = "1d",
            group_by = 'ticker',
            auto_adjust = True,
            prepost = True,
            threads = True,
            proxy = None
        )       

def process_data(data,number_of_stocks,store_file = True):
    all_data_sql = []

    for symbol in symbols:
        df = pd.DataFrame(data[symbol])
        df.dropna(inplace=True)
        df.reset_index(inplace = True)
        df[['Open','High','Low','Close','Volume']] = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric)
        df.reset_index(drop = True,inplace = True)

        df = calculate_indicators(df)
        df = df[df['Date'] >= '2002-01-02'].reset_index()
        df.drop(columns=['index'], inplace=True)
        df.rename(columns= {'Date':'date'}, inplace=True)
        if store_file:
            df.to_pickle('./data/'+str(symbol)+'.pkl')
        df['symbol'] = symbol
        all_data_sql.extend(df.values)

    total_df = pd.DataFrame(all_data_sql, columns=df.columns)
    grouped = total_df.groupby('date')
    fix_dates = [x for x in grouped.groups if len(grouped.get_group(x)) != number_of_stocks]
    total_df = total_df[~total_df['date'].isin(fix_dates)]
    total_df = total_df.sort_values(by=['date','symbol'],ascending=True)
    total_df.drop(columns=['symbol'],inplace=True) 
    total_df['date'] = total_df['date'].astype(str)
    
    return total_df

if __name__=='__main__':
    symbols = ['AAPL','CSCO','INTC','ORCL','MSFT','IBM','HON','VZ','MFC','JPM','BAC','TD','MMM','CAT','BA','GE','WMT','KO', 'HD','AMZN','JNJ','MRK','PFE','GILD','ENB','CVX','BP','RDSB.L']
    data = download_data(symbols)
    prepeared_df = process_data(data,number_of_stocks=len(symbols))
    db_create("postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski", prepeared_df)
            

