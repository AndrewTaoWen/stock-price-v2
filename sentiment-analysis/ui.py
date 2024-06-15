import streamlit as st
import data
import lstm
import os

def main():
    st.title("Stock Price Prediction with Sentiment Analysis")
    
    stock_ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    api_key = os.environ['x_api_key']
    api_secret_key = os.environ['x_api_secret_key']
    access_token = os.environ['x_access_token']
    access_token_secret = os.environ['x_access_token_secret']
    
    if st.button("Predict"):
        # Fetch and process data
        stock_data = data.get_stock_data(stock_ticker, start_date, end_date)
        twitter_data = data.get_twitter_data(api_key, api_secret_key, access_token, access_token_secret, stock_ticker)
        news_data = data.get_news_data(api_key, stock_ticker, start_date, end_date)
        
        sentiment_scores = data.analyze_sentiment(twitter_data + news_data)
        X, y, scaler = lstm.prepare_data(stock_data, sentiment_scores)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = lstm.train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build, train and evaluate model
        model = lstm.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model = lstm.train_model(model, X_train, y_train)
        predictions = lstm.make_predictions(model, X_test, scaler)
        
        # Display results
        st.line_chart(stock_data['Close'])
        st.line_chart(predictions)
        
if __name__ == "__main__":
    main()
