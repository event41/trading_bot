import backtrader as bt
from strategy import MyStrategy
from datetime import datetime
from dotenv import load_dotenv
import os
import argparse
import logging
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle

# Настройка логирования
logging.basicConfig(filename='optimize.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Загрузка переменных окружения из файла .env
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Optimization script')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for optimization in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date for optimization in YYYY-MM-DD format')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--leverage', type