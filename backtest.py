from __future__ import absolute_import, division, print_function, unicode_literals
import backtrader as bt
import pandas as pd
from strategy import MyStrategy
from keras.models import load_model
import pickle
from tensorflow.keras.models import load_model  # Если вы используете Keras

# Загрузка данных
data = bt.feeds.GenericCSVData(
    dataname='btc1hdata.csv',  # Укажите путь к вашему файлу данных
    dtformat='%Y-%m-%d %H:%M:%S',  # Формат даты в файле
    datetime=0,  # Индекс столбца с датой
    open=1,      # Индекс столбца с ценой открытия
    high=2,      # Индекс столбца с максимальной ценой
    low=3,       # Индекс столбца с минимальной ценой
    close=4,     # Индекс столбца с ценой закрытия
    volume=5,    # Индекс столбца с объемом
    openinterest=-1  # Открытый интерес (если нет, ставим -1)
)

# Создание экземпляра Cerebro
cerebro = bt.Cerebro()

# Добавление данных
cerebro.adddata(data)

# Загрузка модели (например, обученная LSTM или другая ML-модель)
try:
    model = load_model('lstm_model-1.h5')  # Укажите путь к вашей модели
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None  # Если модель не загружена, устанавливаем значение None

# Добавление стратегии с моделью
cerebro.addstrategy(MyStrategy, model=model)

# Настройка начального капитала
cerebro.broker.set_cash(1000.0)

# Настройка комиссий (если необходимо)
cerebro.broker.setcommission(commission=0.15)  # Комиссия 0.1%

# Запуск бэктеста
print(f"Начальный капитал: {cerebro.broker.getvalue():.2f}")
results = cerebro.run()
print(f"Конечный капитал: {cerebro.broker.getvalue():.2f}")

# Визуализация результатов (опционально)
cerebro.plot(style='candlestick')
