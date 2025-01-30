import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import pickle

# Путь к данным
data_path = 'history/BTC_USDT_1h_historical_data.csv'

# Загрузка данных
data = pd.read_csv(data_path)

# Выбор признаков и целевой переменной
features = ['close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal', 'ATR', 'CCI']
target = 'close'

X = data[features].values
y = data[target].values.reshape(-1, 1)

# Нормализация данных
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Создание последовательностей для LSTM
def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Создание модели LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Обучение модели
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Сохранение модели и скалера
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/lstm_model.h5')
with open('models/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Модель обучена и сохранена!")

import matplotlib.pyplot as plt

# Получение истории обучения
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Построение графика ошибок
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
