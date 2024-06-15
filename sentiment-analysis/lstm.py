import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def prepare_data(stock_data, sentiment_scores):
    stock_data['sentiment'] = sentiment_scores
    data = stock_data[['Close', 'sentiment']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(50, len(scaled_data)):
        X.append(scaled_data[i-50:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=200, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=150))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
