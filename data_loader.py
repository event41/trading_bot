import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
import talib
from datetime import datetime

# Загрузка переменных окружения из файла .env
load_dotenv()

class DataLoader:
    def __init__(self, api_key, secret):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'options': {
                'recvWindow': 10000  # Увеличиваем recvWindow до 10 секунд
            }
        })

    def fetch_historical_data(self, symbol, timeframe, start_date, end_date):
        since = int(start_date.timestamp() * 1000)  # Преобразуем дату в миллисекунды
        limit = 1000  # Максимальное количество свечей за один запрос
        all_ohlcv = []

        while since < int(end_date.timestamp() * 1000):
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break  # Если больше данных нет, выходим из цикла
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1  # Обновляем временную метку для следующего запроса

        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

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

if __name__ == '__main__':
    api_key = os.getenv('API_KEY')
    secret = os.getenv('SECRET')
    loader = DataLoader(api_key, secret)
    
    pairs = ['BTC/USDT', 'XRP/USDT', 'DOGE/USDT', 'SEI/USDT']
    timeframes = ['1h', '30m', '15m']
    start_date = datetime.strptime(os.getenv('START_DATE'), '%Y-%m-%d')
    end_date = datetime.strptime(os.getenv('END_DATE'), '%Y-%m-%d')

    # Создаем папку history, если она не существует
    if not os.path.exists('history'):
        os.makedirs('history')

    for pair in pairs:
        for timeframe in timeframes:
            try:
                df = loader.fetch_historical_data(pair, timeframe, start_date, end_date)
                df = add_indicators(df)
                
                # Сохраняем файл в папку history
                file_path = f'history/{pair.replace("/", "_")}_{timeframe}_historical_data.csv'
                df.to_csv(file_path, index=False)
                print(f"Data for {pair} on {timeframe} loaded and saved to {file_path}.")
            except Exception as e:
                print(f"Error fetching data for {pair} on {timeframe}: {e}")
