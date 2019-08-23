import urllib.request
import json
import datetime
import pandas as pd
import pandas_datareader.data as web
from pandas.tseries.offsets import BDay


apiKey = "79f83d1f-f210-4c8f-81e7-2421f5ea63ed"
news_num = 20
days_num = 5000
txt_filename = 'news_titles.txt'
raw_filename = 'raw_data.csv'

stockIdx_filename = 'sp500.csv'

start_date = datetime.datetime(2005, 8, 22)


def extract_news_titles(api_key, start_date, n_days, article_num, out_filename):
    date = start_date
    
    for i in range(n_days):
        
        date += datetime.timedelta(days=1)
        
        dateStrDash = date.strftime('%Y-%m-%d')
        dateStrSlash = date.strftime('%Y/%m/%d')
        
        url = 'http://content.guardianapis.com/search?from-date={0}&to-date={1}&page-size=25&api-key={2}'.format(
                dateStrDash, dateStrDash, apiKey)
        
        output = dateStrSlash
        try:
            # Convert from bytes to text
            resp_text = urllib.request.urlopen(url).read().decode('UTF-8')
            # Use loads to decode from text to a python object
            json_obj = json.loads(resp_text)
            
            index = 0
            while (index < article_num):
                try:
                    output = output + '\t' + (json_obj['response']['results'][index]['webTitle'])
                finally:
                    index = index + 1
            output = output + '\n'
            print(output)
            
            with open(out_filename, 'a') as myfile:
                myfile.write(output)
        except:
            pass
            

def build_news_df_from_txt(out_filename):
    with open(out_filename) as f:
        news = f.read().splitlines()
        
    news = [line.split('\t') for line in news]
    news = [line for line in news if len(line)==21] # Filter out the missing data
    
    # Construct a dataframe for the news titles, with date as index
    news_dict = {content[0]:content[1:] for content in news}
    news_df = pd.DataFrame.from_dict(news_dict).T
    news_df.index = [datetime.datetime.strptime(dateStr, '%Y/%m/%d') for dateStr in news_df.index]
    news_df.columns = ['Top {}'.format(i+1) for i in range(news_num)]
    return news_df


def extract_stockIdx_prices(news_df, start_dt, n_days, idx_ticker='^GSPC', source='yahoo'):
    start_dt = start_dt + datetime.timedelta(days=1)
    end_dt = start_dt + datetime.timedelta(days=n_days)
    
    idx_series = web.DataReader(idx_ticker, source, start_dt, end_dt)['Adj Close']
    
    news_dates = news_df.index
    price_dates = list(news_dates) + [news_dates[-1] + BDay(1)]
    price_dates = pd.DatetimeIndex(price_dates)
    
    idx_prices = idx_series[price_dates]
    return idx_prices
    

def read_stockIdx_prices(stockIdx_filename, news_df):
    idx_df = pd.read_csv(stockIdx_filename, index_col=0)['Adj Close']
    
    news_dates = news_df.index
    price_dates = [news_date.strftime('%Y-%m-%d') for news_date in news_dates] + [
            (news_dates[-1] + BDay(1)).strftime('%Y-%m-%d')]
    
    idx_prices = idx_df[price_dates]
    return idx_prices


def gen_flucturation_label(idx_prices):
    idx_prices = pd.Series(idx_prices)
    
    idx_diffs = idx_prices.diff()
    
    # Adjust the index, to let today's date correpond to its future price fluctuation
    news_dates = idx_diffs.index[:-1]
    idx_diffs = idx_diffs[1:]
    idx_diffs.index = news_dates
    
    idx_diffs = idx_diffs.dropna()  # Filter out invalid data
    
    # Label the data as 1 if the future price goes up, otherwise label as 0
    idx_labels = idx_diffs.map(lambda x: 1 if x > 0 else 0)
    
    return idx_labels
    

if __name__ == '__main__':
    '''
    extract_news_titles(apiKey, start_date, days_num, news_num, txt_filename)
    '''
    news_df = build_news_df_from_txt(txt_filename)
    
    # Get stock index prices:
    # either extract through API, or download the data as a csv then read it
    idx_prices = extract_stockIdx_prices(news_df, start_date, days_num, idx_ticker='^DJI', source='yahoo')
    #idx_prices = read_stockIdx_prices(stockIdx_filename, news_df)
    idx_labels = gen_flucturation_label(idx_prices)
    
    # Merge news_df and the labels on the dates
    news_and_prices = news_df.join(idx_labels, how='right')
    news_and_prices.columns = list(news_and_prices.columns)[:-1] + ['Label']
    news_and_prices.to_csv(raw_filename)
