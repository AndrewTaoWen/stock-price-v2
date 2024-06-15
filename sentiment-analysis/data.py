import tweepy
import os
import urllib
import json
import pandas as pd
import datetime as dt


def get_stock_data(ticker, start_date, end_date):
    # Get API key
    stock_api_key = os.environ["alphavantage_api_key"]

    # URL to fetch historical stock data
    url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={stock_api_key}"

    # Save data to this file
    file_to_save = f'stock_market_data-{ticker}.csv'

    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # Extract stock market data
            data = data['Time Series (Daily)']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                            float(v['4. close']), float(v['1. open'])]
                df.loc[-1, :] = data_row
                df.index = df.index + 1
        df.to_csv(file_to_save, index=False)
        print(f'Data saved to: {file_to_save}')
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the DataFrame according to the start_date and end_date
    mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    filtered_df = df.loc[mask]

    return filtered_df

def get_twitter_data(api_key, api_secret_key, access_token, access_token_secret, query, count=100):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret_key, access_token, access_token_secret)
    api = tweepy.API(auth)
    tweets = api.search(q=query, count=count, lang='en')
    return [tweet.text for tweet in tweets]

from newsapi import NewsApiClient

def get_news_data(api_key, query, from_date, to_date):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, from_param=from_date, to=to_date, language='en')
    return [article['title'] + " " + article['description'] for article in articles['articles']]

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [analyzer.polarity_scores(text)['compound'] for text in texts]
    return sentiment_scores
