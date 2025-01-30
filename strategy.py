from keras.models import load_model
import pickle

# Загрузка модели и скалеров
model = load_model('models/lstm_model.h5')
with open('models/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('models/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

import pandas as pd
import talib
import backtrader as bt
from datetime import datetime
import logging
import psycopg2
from dotenv import load_dotenv
import os
import pickle
from keras.models import load_model
import numpy as np

# Настройка логирования
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Настройка подключения к базе данных
def connect_db():
    DB_NAME = 'trading_bot'
    DB_USER = 'botuser'
    DB_PASSWORD = 'yourpassword'
    DB_HOST = 'localhost'
    DB_PORT = 5432
    conn = psycopg2.connect(f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}")
    return conn

def add_indicators(df):
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    macd, macdsignal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['Signal'] = macdsignal
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)

    # Bollinger Bands
    upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['Upper_Band'] = upper_band
    df['Middle_Band'] = middle_band
    df['Lower_Band'] = lower_band

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

    # Ичимоку Клоба (ручное вычисление)
    df['Tenkan_Sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['close'].shift(-26)

    # Дополнительные источники данных
    df['hl2'] = (df['high'] + df['low']) / 2
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['hlcc4'] = (df['high'] + df['low'] + df['close'] + df['close']) / 4
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Нестандартные индикаторы
    df['Williams_%R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    df['MOM'] = talib.MOM(df['close'], timeperiod=10)
    df['Force_Index'] = df['close'].diff() * df['volume']
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['Chaikin_Money_Flow'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['DPO'] = df['close'] - df['close'].rolling(window=20).mean().shift(11)  # Ручное вычисление DPO
    df['ROC'] = talib.ROC(df['close'], timeperiod=10)
    
    return df.dropna()

def calculate_risk_management(df, current_price, predicted_price, is_long=True):
    atr = df['ATR'].iloc[-1]
    if is_long:
        stop_loss = current_price - atr * 2
        take_profit = current_price + atr * 3
        trailing_stop = current_price - atr * 1
    else:
        stop_loss = current_price + atr * 2
        take_profit = current_price - atr * 3
        trailing_stop = current_price + atr * 1
    return stop_loss, take_profit, trailing_stop

class MyStrategy(bt.Strategy):
    params = (
        ('sma_period_50', 50),
        ('sma_period_200', 200),
        ('take_profit_levels', [0.01, 0.02, 0.03]),
        ('break_even_threshold', 0.01),
        ('averaging_levels', [0.02, 0.04, 0.06]),
        ('model', None),  # Передаем обученную модель как параметр
        ('scaler', None),  # Передаем скалер как параметр
        ('max_drawdown_limit', None),  # Максимально допустимая просадка
        ('position_percent', 5),  # Фиксированный процент от депозита для каждой позиции
    )

# Реализация индикатора VWAP
class VWAP(bt.Indicator):
    lines = ('vwap',)
    params = (('period', 20),)

    def __init__(self):
        # Вычисляем типичную цену (среднее между high, low, close)
        typical_price = (self.data.high + self.data.low + self.data.close) / 3

        # Умножаем типичную цену на объем
        tp_volume = typical_price * self.data.volume

        # Скользящая сумма типичной цены * объем
        rolling_tp_volume = bt.indicators.SMA(tp_volume, period=self.params.period) * self.params.period

        # Скользящая сумма объема
        rolling_volume = bt.indicators.SMA(self.data.volume, period=self.params.period) * self.params.period

        # Рассчитываем VWAP
        self.lines.vwap = rolling_tp_volume / rolling_volume

# Основная стратегия
class MyStrategy(bt.Strategy):
    params = (
        ('sma_period_50', 50),
        ('sma_period_200', 200),
        ('model', None),  # Добавляем параметр model
    )

    def __init__(self):
        # Добавляем индикаторы SMA
        self.sma_50_1h = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.sma_period_50
        )
        self.sma_200_1h = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.sma_period_200
        )

        # Добавляем индикатор RSI
        self.rsi_1h = bt.indicators.RSI(self.datas[0].close, period=14)

        # Добавляем индикатор MACD
        macd = bt.indicators.MACD(self.datas[0].close)
        self.macd_1h = macd.macd  # Линия MACD
        self.signal_1h = macd.signal  # Линия сигнала

        # Добавляем индикатор ATR
        self.atr_1h = bt.indicators.ATR(self.datas[0], period=14)

        # Добавляем индикатор CCI
        self.cci_1h = bt.indicators.CCI(self.datas[0], period=20)

        # Добавляем индикатор VWAP
        self.vwap_1h = VWAP(self.datas[0], period=20)

        # Bollinger Bands
        self.bbands_1h = bt.indicators.BollingerBands(self.datas[0].close, period=20, devfactor=2)
        self.upper_band_1h = self.bbands_1h.top
        self.middle_band_1h = self.bbands_1h.mid
        self.lower_band_1h = self.bbands_1h.bot

        # Ичимоку Клоба (ручное вычисление)
        self.tenkan_sen_1h = (bt.indicators.Highest(self.datas[0].high, period=9) + bt.indicators.Lowest(self.datas[0].low, period=9)) / 2
        self.kijun_sen_1h = (bt.indicators.Highest(self.datas[0].high, period=26) + bt.indicators.Lowest(self.datas[0].low, period=26)) / 2
        self.senkou_span_a_1h = ((self.tenkan_sen_1h + self.kijun_sen_1h) / 2)(-26)  # Задержка на 26 периодов
        self.senkou_span_b_1h = ((bt.indicators.Highest(self.datas[0].high, period=52) + bt.indicators.Lowest(self.datas[0].low, period=52)) / 2)(-26)  # Задержка на 26 периодов
        self.chikou_span_1h = self.datas[0].close(-26)  # Задержка на 26 периодов

        # Дополнительные источники данных
        self.hl2_1h = (self.datas[0].high + self.datas[0].low) / 2
        self.hlc3_1h = (self.datas[0].high + self.datas[0].low + self.datas[0].close) / 3
        self.hlcc4_1h = (self.datas[0].high + self.datas[0].low + self.datas[0].close + self.datas[0].close) / 4
        self.ohlc4_1h = (self.datas[0].open + self.datas[0].high + self.datas[0].low + self.datas[0].close) / 4

        # Нестандартные индикаторы
        self.williams_r_1h = bt.indicators.WilliamsR(self.datas[0].high, self.datas[0].low, self.datas[0].close, period=14)
        self.stoch_1h = bt.indicators.Stochastic(self.datas[0].high, self.datas[0].low, self.datas[0].close, period=14, period_dfast=3, period_slow=3)
        self.stoch_k_1h = self.stoch_1h.percK
        self.stoch_d_1h = self.stoch_1h.percD
        self.ema_50_1h = bt.indicators.ExponentialMovingAverage(self.datas[0].close, period=50)
        self.mom_1h = bt.indicators.Momentum(self.datas[0].close, period=10)
        self.force_index_1h = self.datas[0].close - self.datas[0].close(-1) * self.datas[0].volume
        self.obv_1h = bt.indicators.OnBalanceVolume(self.datas[0].close, self.datas[0].volume)
        self.chaikin_money_flow_1h = bt.indicators.ChaikinMoneyFlow(self.datas[0].high, self.datas[0].low, self.datas[0].close, self.datas[0].volume, fastperiod=3, slowperiod=10)
        self.adx_1h = bt.indicators.ADX(self.datas[0].high, self.datas[0].low, self.datas[0].close, period=14)
        self.dpo_1h = self.datas[0].close - bt.indicators.SimpleMovingAverage(self.datas[0].close(-11), period=20)  # Ручное вычисление DPO
        self.roc_1h = bt.indicators.RateOfChange(self.datas[0].close, period=10)

        # Сохраняем модель
        self.model = self.params.model

    def next(self):
        # Пример использования модели для прогнозирования
        if self.model:
            # Получаем последние данные для прогнозирования
            last_data = [self.datas[0].close[0], self.rsi_1h[0], self.macd_1h[0]]
            prediction = self.model.predict([last_data])[0]

            # Принимаем решение на основе прогноза
            if prediction > 0:
                print("Прогноз: цена вырастет - покупка")
            else:
                print("Прогноз: цена упадет - продажа")

    # Пример использования VWAP в стратегии
    if self.datas[0].close[0] > self.vwap_1h[0]:
        print("Цена выше VWAP - покупка")
    elif self.datas[0].close[0] < self.vwap_1h[0]:
        print("Цена ниже VWAP - продажа")

    # Используем данные с 1 часа для принятия решений
    stop_loss, take_profit, trailing_stop = calculate_risk_management(
        pd.DataFrame({
            'high': self.datas[0].high.get(ago=0, size=1)[0],
            'low': self.datas[0].low.get(ago=0, size=1)[0],
            'close': self.datas[0].close.get(ago=0, size=1)[0]
        }),
        self.datas[0].close[0],
        self.datas[0].close[0],
        is_long=True if self.position.size >= 0 else False
    )

    # Получаем последние 60 значений для предсказания цены
    last_60_values = self.datas[0].get(size=60)
    last_60_df = pd.DataFrame(last_60_values, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    last_60_df = add_indicators(last_60_df)

    # Нормализуем данные
    X_last = self.params.scaler.transform(
        last_60_df[['close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Signal', 'ATR', 'CCI',
                    'Upper_Band', 'Middle_Band', 'Lower_Band', 'VWAP',
                    'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span',
                    'hl2', 'hlc3', 'hlcc4', 'ohlc4',
                    'Williams_%R', 'Stoch_K', 'Stoch_D', 'EMA_50', 'MOM',
                    'Force_Index', 'OBV', 'Chaikin_Money_Flow', 'ADX', 'DPO', 'ROC']].values
    )
    X_last = np.reshape(X_last, (1, X_last.shape[0], X_last.shape[1]))

    # Предсказываем цену
    predicted_price = self.params.model.predict(X_last)
    predicted_price = self.params.scaler.inverse_transform(
        np.concatenate((X_last[0, -1, 1:].reshape(1, -1), predicted_price), axis=1)
    )[:, -1][0]

    # Расчет размера позиции на основе процента от депозита
    position_size = self.broker.getvalue() * self.params.position_percent / self.datas[0].close[0]

    # Условия для входа в LONG позицию
    if (self.sma_50_1h[0] > self.sma_200_1h[0] and
        self.rsi_1h[0] < 70 and
        self.macd_1h[0] > self.signal_1h[0] and
        self.cci_1h[0] < 100):

        if predicted_price > self.datas[0].close[0]:
            self.buy(exectype=bt.Order.Limit, price=self.datas[0].close[0], size=position_size)
            self.opened_trades.append({'price': self.datas[0].close[0], 'size': position_size, 'stop_loss': stop_loss})

            # Предсказываем параметры тейк-профита, стоп-лосса и усреднения
            if self.params.model:
                prediction = self.params.model.predict([[self.datas[0].close[0], self.datas[0].close[0]]])[0]
                take_profit_levels = prediction[:3]
                stop_loss_level = prediction[3]
                averaging_levels = prediction[4:]
            else:
                take_profit_levels = self.params.take_profit_levels
                stop_loss_level = stop_loss
                averaging_levels = self.params.averaging_levels

            # Устанавливаем уровни тейк-профита
            for i, level in enumerate(take_profit_levels):
                take_profit_price = self.datas[0].close[0] * (1 + level) if self.position.size >= 0 else self.datas[0].close[0] * (1 - level)
                self.sell(size=self.position.size / len(take_profit_levels), exectype=bt.Order.Limit, price=take_profit_price)

            # Устанавливаем стоп-лосс
            self.sell(
                size=self.position.size,
                exectype=bt.Order.Stop,
                price=self.datas[0].close[0] * (1 - stop_loss_level) if self.position.size >= 0 else self.datas[0].close[0] * (1 + stop_loss_level)
            )

            # Устанавливаем уровни усреднения
            for i, level in enumerate(averaging_levels):
                averaging_price = self.datas[0].close[0] * (1 - level) if self.position.size >= 0 else self.datas[0].close[0] * (1 + level)
                self.buy(size=1, exectype=bt.Order.Stop, price=averaging_price, oco=self.sell_order)

    # Условия для входа в SHORT позицию
    elif (self.sma_50_1h[0] < self.sma_200_1h[0] and
          self.rsi_1h[0] > 30 and
          self.macd_1h[0] < self.signal_1h[0] and
          self.cci_1h[0] > -100):

        if predicted_price < self.datas[0].close[0]:
            self.sell(exectype=bt.Order.Limit, price=self.datas[0].close[0], size=position_size)
            self.opened_trades.append({'price': self.datas[0].close[0], 'size': -position_size, 'stop_loss': stop_loss})

            # Предсказываем параметры тейк-профита, стоп-лосса и усреднения
            if self.params.model:
                prediction = self.params.model.predict([[self.datas[0].close[0], self.datas[0].close[0]]])[0]
                take_profit_levels = prediction[:3]
                stop_loss_level = prediction[3]
                averaging_levels = prediction[4:]
            else:
                take_profit_levels = self.params.take_profit_levels
                stop_loss_level = stop_loss
                averaging_levels = self.params.averaging_levels

            # Устанавливаем уровни тейк-профита
            for i, level in enumerate(take_profit_levels):
                take_profit_price = self.datas[0].close[0] * (1 - level) if self.position.size <= 0 else self.datas[0].close[0] * (1 + level)
                self.buy(size=abs(self.position.size) / len(take_profit_levels), exectype=bt.Order.Limit, price=take_profit_price)

            # Устанавливаем стоп-лосс
            self.buy(
                size=abs(self.position.size),
                exectype=bt.Order.Stop,
                price=self.datas[0].close[0] * (1 + stop_loss_level) if self.position.size <= 0 else self.datas[0].close[0] * (1 - stop_loss_level)
            )

            # Устанавливаем уровни усреднения
            for i, level in enumerate(averaging_levels):
                averaging_price = self.datas[0].close[0] * (1 + level) if self.position.size <= 0 else self.datas[0].close[0] * (1 - level)
                self.sell(size=1, exectype=bt.Order.Stop, price=averaging_price, oco=self.buy_order)

    def stop(self):
        # Проверяем максимальную просадку
        max_drawdown = self.analyzers.drawdown.get_analysis()['max']['drawdown']
        print(f"Max Drawdown: {max_drawdown:.2f}%")

        # Расчет доходности
        start_value = self.broker.startingcash
        end_value = self.broker.getvalue()
        total_return = ((end_value - start_value) / start_value) * 100
        print(f"Total Return: {total_return:.2f}%")

        # Расчет коэффициента Шарпа
        returns = self.analyzers.returns.get_analysis()['rtot']
        std_dev = self.analyzers.returns.get_analysis()['stddev']
        if std_dev != 0:
            sharpe_ratio = returns / std_dev
        else:
            sharpe_ratio = 0
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Логирование результатов
        logging.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logging.info(f"Total Return: {total_return:.2f}%")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
