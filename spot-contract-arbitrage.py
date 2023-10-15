# -*- coding: utf-8 -*-
# Copyright (C) hychien.tw @ MurMurCats NFT Society

import os
import sys
import json
from pathlib import Path
import time
from datetime import timedelta
import tkinter as tk
from tkinter import ttk, font
import requests

import numpy as np
import pandas as pd
import threading
from functools import partial

from pybit.unified_trading import HTTP

# modify the API key and secret here
API_KEY = ''
API_SECRET = ''

# transaction logs will be stored in this directory for cache use
FILE_DIR = Path(__file__).parent
LOG_DIR = FILE_DIR / 'transaction-logs'

DEBUG_MODE = os.environ.get('DEBUG', False)
_DAY = 86400000
_HOUR = 3600000
_MINUTE = 60000
DEFAULT_DIGIT = 12
DEFAULT_PRECISION = 4

INTERVALS = {_MINUTE: '1', _HOUR: '60', _DAY: 'D'}

STABLE_COINS = ['USDT', 'USDC']
WALLET_COLUMNS = dict(
    symbol='貨幣',
    balance='現貨淨值',
    spot_pnl='現貨盈虧',
    position='合約持倉',
    fr='資金費率',
    borrow_rate='借貸利率',
    cumrealpnl='已實現盈虧',
    unrlpnl='未實現盈虧',
    total_pnl='總盈虧 (U)',
    spot_price='現貨價格',
    contract_price='合約價格',
    close_price_diff='平倉價差 (負優)',
)
COLUMNS_WIDTH = {name: 74 for name in WALLET_COLUMNS.keys()}
COLUMNS_WIDTH['close_price_diff'] = 90


def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def get_local_time(time_stamp, unit=0.001):
    if isinstance(time_stamp, str):
        time_stamp = int(time_stamp)
    time_stamp = time_stamp * unit
    struct_time = time.localtime(time_stamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", struct_time)


def get_time_stamp():
    return int(time.mktime(time.localtime())) * 1000


def miniseconds_to_datetime(miniseconds, format='%H:%M:%S'):
    td = timedelta(seconds=int(miniseconds / 1000))
    time_obj = time.gmtime(td.total_seconds())
    return time.strftime(format, time_obj)


def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False
    except TypeError:  # like None
        return False


def to_float(val, default=None):
    """_summary_
        In some cases, the value is None or '', or the value is a string of float.
    """
    if is_float(val):
        return float(val)
    return default


def base_precision_filter(quantity: float, base_precision: float, ceil: bool = False):
    # return int(quantity / base_precision) * base_precision
    sign = 1 if quantity >= 0 else -1
    if ceil:
        return sign * np.ceil(sign * quantity / base_precision) * base_precision
    return sign * round(sign*quantity, int(-np.log10(base_precision)))


def get_precision(num_str):
    # return len(str(float(num_str)).split('.')[-1])
    return int(-np.log10(float(num_str)))


def get_contract_symbol(coin, currency='USDT'):
    if coin in STABLE_COINS:
        return coin
    return f'{coin}{currency}'


def position_size_with_sign(size, side):
    if size == 0 and side == '':
        return 0
    assert side in ['Buy', 'Sell'], f'Only But and Sell sides, but got {side}.'
    if isinstance(size, str):
        size = float(size)
    if side == 'Buy':
        return size
    return -size


class APIExecutor:
    def __init__(self):
        self.session = HTTP(api_key=API_KEY, api_secret=API_SECRET)

        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)

        self.cache = dict(
            linear=self.log_dir / f'{API_KEY}-linear.csv',
            spot=self.log_dir / f'{API_KEY}-spot.csv',
            usdcusdt_exchange=self.log_dir / f'{API_KEY}-usdcusdt.csv',
            instruments_info=self.log_dir / f'instruments-info.json',
            borrow_history=self.log_dir / f'{API_KEY}-borrow-history.csv',
        )
        self.linear_log_df = self.load_file(self.cache['linear'])
        self.spot_log_df = self.load_file(self.cache['spot'])
        self.usdc_exchange_log_df = self.load_file(self.cache['usdcusdt_exchange'])
        self.instruments_info = self.load_file(self.cache['instruments_info'])
        self.borrow_history_df = self.load_file(self.cache['borrow_history'])

        self.trans_log_df = dict(spot=self.spot_log_df, linear=self.linear_log_df)

        self.k_line_tags = [
            'startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice',
            'volume', 'turnover'
        ]

    @staticmethod
    def load_file(file_path):
        if Path(file_path).suffix == '.csv':
            return pd.read_csv(file_path) if Path(file_path).exists() else None
        if Path(file_path).suffix == '.json':
            return json.load(open(
                file_path, mode='r')) if Path(file_path).exists() else dict()
        raise TypeError(
            f'Unsupported File extension: {Path(file_path).suffix}. Should be JSON or CSV.'
        )

    def get_instruments_info(self, category, symbol):
        """_summary_
            Get the SPEC of symbol, including lotSizeFilter, priceFilter, leverageFilter.
            Also store them in the file cache.
        """
        if f'{category}:{symbol}' in self.instruments_info:
            return self.instruments_info[f'{category}:{symbol}']
        instruments_info = self.session.get_instruments_info(
            category=category, symbol=symbol)['result']['list']
        if len(instruments_info) > 0:
            self.instruments_info[f'{category}:{symbol}'] = dict(
                **instruments_info[0]['lotSizeFilter'],
                **instruments_info[0]['priceFilter'],
                **instruments_info[0].get('leverageFilter', {}),
            )
            if category == 'spot':
                self.instruments_info[f'{category}:{symbol}']['marginTrading'] = instruments_info[0].get(
                    'marginTrading', False)
            json.dump(self.instruments_info, open(self.cache['instruments_info'], mode='w'), indent=4)
            return self.instruments_info[f'{category}:{symbol}']
        return {}

    @property
    def wallet(self):
        return self.session.get_wallet_balance(accountType='UNIFIED')['result']['list'][0]

    def get_wallet_margin_rate(self):
        wallet = self.wallet
        return dict(IMRate=wallet['accountIMRate'],
                    MMRate=wallet['accountMMRate'],
                    )

    def get_available_balance(self):
        return float(self.wallet['totalAvailableBalance'])

    def get_wallet_balance(self):
        wallet = self.wallet
        # json.dump(wallet, open('wallet.json', mode='w'), ensure_ascii=False, indent=4)
        return {
            coin_dict['coin']: coin_dict['equity']
            for coin_dict in wallet['coin']
            if abs(float(coin_dict['usdValue'])) > 0.1
            or coin_dict['coin'] in STABLE_COINS
        }

    def get_spot_balance(self, coin):
        wallet = self.session.get_wallet_balance(
            accountType='UNIFIED', coin=coin)['result']['list'][0]
        return to_float(wallet['coin'][0]['walletBalance'], default=0)

    def start_time_filter(self, dSeries, start_time, time_interval=_DAY, front=True):
        if front:
            return dSeries.apply(lambda x: int(x) <= int(start_time) < int(x) + time_interval)
        if time_interval is not None:
            return dSeries.apply(lambda x: int(start_time) <= int(x) < int(start_time) + time_interval)
        else:
            return dSeries.apply(lambda x: int(start_time) <= int(x))

    def get_usdc_historical_price(self,
                                  start_time,
                                  key='highPrice',
                                  time_interval=_MINUTE):
        _time_interval_symbol = INTERVALS[time_interval]
        df = self.usdc_exchange_log_df
        if df is not None:
            result = df[self.start_time_filter(df.startTime,
                                               start_time,
                                               time_interval=time_interval)]
            if len(result):
                return to_float(result[key].iloc[0])
        k_line = self.session.get_kline(
            category='spot',
            symbol='USDCUSDT',
            interval=_time_interval_symbol,
            start=str(int(start_time) - time_interval),
            limit=1)
        result = {key: value for key, value in zip(self.k_line_tags, k_line['result']['list'][0])}
        self.usdc_exchange_log_df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        self.usdc_exchange_log_df.to_csv(self.cache['usdcusdt_exchange'], index=False)
        return float(result[key])

    def get_contract_ask1_price(self, symbol):
        return float(self.get_contract_tickers(symbol)['ask1Price'])

    def get_contract_bid1_price(self, symbol):
        return float(self.get_contract_tickers(symbol)['bid1Price'])

    def get_contract_tickers(self, symbol):
        return self.get_tickers(symbol, category='linear')

    def get_price_to_fulfill_amount(self, symbol, amount):
        orderbook = self.session.get_orderbook(category='spot',
                                               symbol=symbol,
                                               limit=50)
        if amount > 0:
            orderbook = orderbook['result']['a']
        else:
            orderbook = orderbook['result']['b']
            amount = -amount
        for order in orderbook:
            amount -= float(order[1])
            if amount <= 0:
                return float(order[0])
        return float(orderbook[-1][0])

    def get_funding_rate(self, symbol):
        tickers = self.get_contract_tickers(symbol)
        return dict(fundingRate=float(tickers['fundingRate']),
                    nextFundingTime=tickers['nextFundingTime'])

    def get_tickers(self, symbol, category='linear'):
        tickers = self.session.get_tickers(
            category=category,
            symbol=symbol,
        )
        return tickers['result']['list'][0]

    def get_spot_tickers(self, symbol):
        return self.get_tickers(symbol, category='spot')

    def get_spot_usd_index_price(self, symbol):
        return float(self.get_spot_tickers(symbol)['usdIndexPrice'])

    def get_spot_bid1_price(self, symbol):
        return float(self.get_spot_tickers(symbol)['bid1Price'])

    def get_spot_ask1_price(self, symbol, get_amount=False):
        return float(self.get_spot_tickers(symbol)['ask1Price'])

    def get_positions(self, symbol):
        position = self.session.get_positions(
            category='linear', symbol=symbol)['result']['list'][0]
        return dict(symbol=position['symbol'],
                    side=position['side'],
                    size=to_float(position['size']),
                    avgPrice=to_float(position['avgPrice']),
                    unrealisedPnl=to_float(position['unrealisedPnl']),
                    cumRealisedPnl=to_float(position['cumRealisedPnl']),
                    leverage=position['leverage'],
                    liqPrice=to_float(position['liqPrice']),
                    createdTime=position['createdTime'],
                    updatedTime=position['updatedTime'])

    def get_leverage(self, symbol):
        position = self.session.get_positions(category='linear', symbol=symbol)
        return position['result']['list'][0]['leverage']

    def set_leverage(self, symbol, leverage):
        if self.get_leverage(symbol) == str(leverage):
            return None
        self.session.set_leverage(
            category='linear',
            symbol=symbol,
            buyLeverage=str(leverage),
            sellLeverage=str(leverage),
        )

    def set_spot_margin_mode(self, turn_open=False):
        if turn_open:
            self.session.spot_margin_trade_toggle_margin_trade(spotMarginMode='1')
        else:
            self.session.spot_margin_trade_toggle_margin_trade(spotMarginMode='0')

    def set_spot_margin_leverage(self, leverage):
        self.set_spot_margin_mode(turn_open=True)
        self.session.spot_margin_trade_set_leverage(leverage=leverage)

    def get_min_qty(self, coin, category='linear'):
        if category == 'linear':
            return float(self.get_instruments_info('linear', get_contract_symbol(coin))['minOrderQty'])
        else:
            symbol = get_contract_symbol(coin)
            usdc_symbol = get_contract_symbol(coin, currency='USDC')
            return min(float(self.get_instruments_info('spot', symbol)['minOrderQty']),
                       float(self.get_instruments_info('spot', usdc_symbol).get('minOrderQty', 1e7)))

    def looping_for_transaction_logs(self,
                                     category,
                                     coin,
                                     start_time=None,
                                     cursor=None,
                                     quantity=None):
        """_summary_

        Args:
            category (str): spot or linear (contract)
            coin (str): 需檢索之貨幣
            start_time (str, optional): 本地端存有 cache，可從 cache 中找到最後一筆交易時間，並從該時間開始查詢。
            cursor (str, optional): 翻頁用的指標。
            quantity (str, optional): 當前資產或者倉位的數量，用以計算最早開倉的時間點，並從該時間點開始查詢。
        """
        transaction_logs = self.session.get_transaction_log(
            accountType='UNIFIED',
            category=category,
            currency='USDT' if category == 'linear' else None,
            baseCoin=coin,
            startTime=start_time,
            cursor=cursor,
            limit=50)
        next_page_cursor = transaction_logs['result']['nextPageCursor']
        if quantity is None:
            return transaction_logs['result']['list'], next_page_cursor, None

        to_break_order_id = None  # for spot
        new_transaction_logs = []
        for log in transaction_logs['result']['list']:
            # for spot, there are always 2 logs, one for USDT, one for coin
            # when the quantity is fulfilled, still need to check
            # if following transactions belongs to the same order.
            if category == 'spot':
                log['usdcPrice'] = self.get_usdc_historical_price(
                    log['transactionTime']
                ) if log['currency'] == 'USDC' else np.nan
            if to_break_order_id is not None:
                if log['orderId'] == to_break_order_id:
                    new_transaction_logs.append(log)
                    continue
                break
            new_transaction_logs.append(log)
            if log['type'] != 'TRADE':
                continue
            if category == 'linear':
                quantity -= position_size_with_sign(log['qty'], log['side'])
                if abs(quantity) <= self.get_min_qty(coin, category='linear'):
                    next_page_cursor = None
                    break
            else:  # spot
                borrow_quantity = 0
                if log['currency'] == coin:
                    trans_time = log['transactionTime']
                    borrow_quantity = self.get_borrow_quantity(
                        coin, start_time=trans_time)
                    quantity -= to_float(log['change'])
                if abs(quantity-borrow_quantity) <= self.get_min_qty(coin, category='spot'):
                    next_page_cursor = None
                    to_break_order_id = log['orderId']

        return new_transaction_logs, next_page_cursor, quantity

    def get_borrow_quantity(self, coin, start_time=None):
        borrow_history = self.get_borrow_history_logs(coin)
        if borrow_history is None or len(borrow_history) == 0:
            return 0
        borrow_history = borrow_history[self.start_time_filter(
            borrow_history.createdTime, start_time, time_interval=None, front=False)]
        if len(borrow_history) > 0:
            return borrow_history['borrowCost'].apply(lambda x: float(x)).sum()
        else:
            return 0

    def looping_for_borrow_history(self, coin, cursor=None, start_time=None):
        borrow_history = self.session.get_borrow_history(currency=coin,
                                                         limit=50,
                                                         startTime=start_time,
                                                         cursor=cursor,
                                                         )
        next_page_cursor = borrow_history['result']['nextPageCursor']
        if len(borrow_history['result']['list']) == 0:
            return [], None
        return borrow_history['result']['list'], next_page_cursor

    def get_collateral_info(self, coin):
        collateral_info = self.session.get_collateral_info(currency=coin)
        return collateral_info['result']['list'][0]

    def get_borrow_rate(self, coin):
        collateral_info = self.get_collateral_info(coin)
        return float(collateral_info['hourlyBorrowRate'])

    def get_borrow_history_logs(self, coin):
        first_page = True
        next_page_cursor = None
        my_borrow_history = []
        start_time = None
        df = self.borrow_history_df
        if df is not None and len(df) > 0:
            try:
                start_time = df[(df.currency == coin)].sort_values(
                    'createdTime', ascending=False).iloc[0]['createdTime']
                start_time = str(int(start_time) + 1)
            except IndexError:
                start_time = None
        while first_page or (next_page_cursor is not None):
            first_page = False
            borrow_history, next_page_cursor = self.looping_for_borrow_history(coin=coin,
                                                                               start_time=start_time,
                                                                               cursor=next_page_cursor,
                                                                               )
            my_borrow_history += borrow_history

        # check if update
        if len(my_borrow_history) > 0:
            # update cache
            df = pd.concat([df, pd.DataFrame(my_borrow_history)],
                           ignore_index=True)
            df.createdTime = df.createdTime.apply(lambda x: int(x))
            df = df.sort_values('createdTime', ascending=False)
            df.to_csv(self.cache['borrow_history'], index=False)
            self.borrow_history_df = df

        return df[(df.currency == coin)] if df is not None else None

    def get_transaction_logs(self, category, coin, quantity=None):
        """_summary_

        Args:
            category (str): spot or linear
            coin (str): coin to query
            quantity (str, float, optional):
                if not None, return the latest transaction logs from the position begins,
                i.e. sum over the quantities in transactions until the sum is equal to the quantity.
        """
        first_page = True
        next_page_cursor = None
        my_transaction_logs = []
        start_time = None

        # if the logs are ever cached, retrieve them to get the latest transaction time
        df = self.trans_log_df[category]
        if df is not None and len(df) > 0:
            try:
                start_time = df[(df.symbol == get_contract_symbol(coin)) | (
                    df.symbol == get_contract_symbol(coin, currency='USDC')
                )].sort_values('transactionTime',
                               ascending=False).iloc[0]['transactionTime']
                start_time = str(int(start_time) + 1)
            except IndexError:
                start_time = None

        while first_page or (next_page_cursor is not None):
            first_page = False
            transaction_log, next_page_cursor, quantity = self.looping_for_transaction_logs(
                category=category,
                coin=coin,
                start_time=start_time,
                cursor=next_page_cursor,
                quantity=quantity)
            my_transaction_logs += transaction_log

        # check if update
        if len(my_transaction_logs) > 0:
            # update cache
            df = pd.concat([df, pd.DataFrame(my_transaction_logs)],
                           ignore_index=True)
            df.transactionTime = df.transactionTime.apply(lambda x: int(x))
            df = df.sort_values('transactionTime', ascending=False)
            df.to_csv(self.cache[category], index=False)
            self.trans_log_df[category] = df

        return df[(df.symbol == get_contract_symbol(coin)) |
                  (df.symbol == get_contract_symbol(coin, currency='USDC'))]

    def place_order(self, category, symbol, side, orderType, qty, price):
        args = dict(category=category,
                    symbol=symbol,
                    side=side,
                    orderType=orderType,
                    qty=qty,
                    price=price)
        if category == 'spot':
            args['isLeverage'] = 1
        place_order = self.session.place_order(**args)
        return place_order

    def get_open_orders(self, order_id, category, symbol):
        order = self.session.get_open_orders(category=category,
                                             orderId=order_id,
                                             symbol=symbol)
        return order['result']['list'][0]

    def get_order_history(self, order_id, category, symbol):
        try:
            order_history = self.session.get_order_history(category=category,
                                                           orderId=order_id,
                                                           symbol=symbol)
            return order_history['result']['list'][0]
        except Exception as e:
            return self.get_open_orders(order_id, category, symbol)

    def cancel_order(self, category, symbol, order_id):
        self.session.cancel_order(
            category=category,
            symbol=symbol,
            orderId=order_id,
        )


class ArbitrageExecutor:
    def __init__(self):

        self.api = APIExecutor()
        self.width = 1024
        self.height = 768
        self.wallet_display_frame_height = 400
        self.text_display_frame_height = 100

        self.leverage = 1

        coin_file = FILE_DIR / 'candidate_coins.txt'
        self.coin_list = self.get_coin_list(coin_file)
        self.my_positions = []
        self.coin_to_close = None
        self.digit = DEFAULT_DIGIT
        self.precision = DEFAULT_PRECISION

        # work in progress, 預留版面
        # load processing log
        # processing_cache = f'transaction-logs/{API_KEY}-processing.log'
        # if Path(processing_cache).exists():
        #     with open(processing_cache, mode='r') as fp:
        #         # self.processing_log = fp.read().splitlines()
        #         self.processing_log = fp.read()
        # else:
        #     self.processing_log = ''
        # self.log_fp = open(processing_cache, mode='a')
        self.threads = []

        self.app = tk.Tk()
        self.app.geometry(f'{self.width}x{self.height}')
        self.font = font.Font(family='Courier', size=13)
        self.app_is_killed = False
        self.enable_arbitrage_on_click = False

        self.target = None
        self.target_contract_instrument_info = None
        self.target_spot_instrument_info = None
        self.build_qty = None
        self.update_task = None
        self.use_usdc = False
        self.open_good_side = ''
        self.order_on_action = dict()
        self.continue_to_build = False
        self.display_price_canvas = False
        self.usdc_usd_indexprice = 1

        self.initial_ui()
        self.run_update_price_canvas()

    def get_coin_list(self, coin_file):
        if Path(coin_file).exists():
            with open(coin_file, mode='r') as fp:
                values = fp.read().splitlines()
            return values
        return None

    def initial_ui(self):
        """_summary_
            這裡負責整體版面配置，包含：
            - 控制列：選擇貨幣、輸入建倉數量或 USDT 數量、建倉按鈕、選擇是否使用 USDC 建倉等等
            - 顯示版面一，狀態版面：
                * 建倉買賣價格計算
                * 目前合約與現貨價格比較、資費
            - 顯示版面一之二，倉位維持率：當下建倉 margin rate、當下維持 margin rate
            - 顯示版面二：錢包資訊
            - 控制列二：選擇全平倉貨幣、平倉按鈕
        """
        self.app.title('套利輔助工具')

        self.control_panel_frame = tk.Frame(self.app)
        self.control_panel_frame.grid(row=0, column=0, columnspan=2, padx=0, pady=0, sticky='nw')

        self.calculation_and_processing_frame = tk.Frame(self.app, width=960,)
        self.calculation_and_processing_frame.grid(row=1, column=0, padx=0, pady=0, sticky='nw')
        self.margin_rate_frame = tk.Frame(self.app, height=90)
        self.margin_rate_frame.grid(row=1, column=1, padx=0, pady=0, sticky='nw')

        # work in progress, 預留版面
        # self.scroll_bar_frame = tk.Frame(self.app, width=960, height=30)
        # self.scroll_bar_frame.grid(row=2, column=0, columnspan=2, padx=0, pady=0, sticky='nw')

        self.wallet_frame = tk.Frame(self.app)
        self.wallet_frame.grid(row=3, column=0, columnspan=2, padx=0, pady=0, sticky='nw')

        self.second_control_panel_frame = tk.Frame(
            self.app, width=960, height=30)
        self.second_control_panel_frame.grid(
            row=4, column=0, columnspan=2, padx=0, pady=5, sticky='nw')

        self.init_margin_rate(self.margin_rate_frame)
        self.init_control_panel(self.control_panel_frame)
        self.init_calculation_and_processing_frame(self.calculation_and_processing_frame)
        # self.init_scroll_bar(self.scroll_bar_frame)
        self.init_wallet(self.wallet_frame)
        self.init_second_control_panel(self.second_control_panel_frame)

    def init_second_control_panel(self, frame):

        balance_button = tk.Button(
            frame, text='期現貨再平衡', command=self.balance_spots_with_positions)
        balance_button.grid(row=0, column=3, padx=20)

        label = tk.Label(frame, text='請選擇全平倉貨幣')
        label.grid(row=0, column=0, padx=5)

        # 添加 Combobox 组件到 Frame
        values = self.my_positions

        values.insert(0, '')
        self.combobox = ttk.Combobox(frame, values=values, width=8)
        self.combobox.grid(row=0, column=1, padx=5)
        # 當選擇改變時執行 on_select
        self.combobox.bind('<<ComboboxSelected>>',
                           self.on_select_coin_to_close)

        # 添加 Button 按鈕到 Frame
        button = tk.Button(frame,
                           text='套利平倉',
                           command=self.close_arbitrage_on_click)
        button.grid(row=0, column=2, padx=5)

    def init_calculation_and_processing_frame(self, frame):
        # self.calculation_and_processing_frame = tk.Frame(self.app, width=960, height=200)
        # width 960, height 200

        self.calculation_and_processing_canvas = tk.Canvas(frame, width=900, height=100, bg='#606060')
        self.calculation_and_processing_canvas.pack(padx=0, pady=0)

        self.action = self.calculation_and_processing_canvas.create_text(10, 10, text='', anchor='nw', font=self.font)
        self.processing = self.calculation_and_processing_canvas.create_text(
            10, 40, text='', anchor='nw', font=self.font)

        self.target_price_canvas = tk.Canvas(frame, width=900, height=150, bg='#606060')
        self.target_price_canvas.pack(padx=0, pady=0)
        self.price_diff_info = self.target_price_canvas.create_text(10, 10, text='', anchor='nw', font=self.font)

    def init_scroll_bar(self, frame):
        self.processing_scrollbar = tk.Scrollbar(frame)
        self.processing_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=self.processing_scrollbar.set,
                                   width=110, height=5, bg='#606060', fg='white')
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.processing_scrollbar.config(command=self.text_widget.yview)
        self.text_widget.insert('1.0', self.processing_log)
        self.text_widget.see('1.0')

    def init_margin_rate(self, frame):
        self.margin_rate_text = tk.Label(
            frame, text='', bg='#606060', fg='skyblue', font=self.font)
        self.margin_rate_text.pack(padx=0, pady=5, fill=tk.BOTH, expand=True)

    def update_margin_rate(self):
        margin_rates = self.api.get_wallet_margin_rate()
        margin_rate_text = '保證金率\n\n'
        margin_rate_text += f'IM: {float(margin_rates["IMRate"])*100:5.2f}%\n'
        margin_rate_text += f'MM: {float(margin_rates["MMRate"])*100:5.2f}%'
        self.margin_rate_text.config(text=margin_rate_text)

    def validation(self, amount, is_usdt=False):
        self.enable_arbitrage_on_click = False
        if amount == '' or amount == '-':
            self.update_action_text('')
            return True

        if is_float(amount):
            if not self.target:
                return True
            symbol = get_contract_symbol(self.target)
            # qty needs to be a multiple of minOrderQty
            base_precision = max(float(self.target_spot_instrument_info['basePrecision']),
                                 float(self.target_contract_instrument_info['minOrderQty']))
            if is_usdt:
                self.qty.set('')
                price = self.api.get_spot_ask1_price(symbol)
                qty = base_precision_filter(float(amount) / price, base_precision)
            else:
                self.usdt_amount.set('')
                qty = base_precision_filter(float(amount), base_precision)

            collateral_info = self.api.get_collateral_info(self.target)
            msg = ''
            if qty < 0:
                if not collateral_info['borrowable']:
                    self.update_action_text(f'{self.target} 無法借貸，無法建倉')
                    return True
                borrow_amount = min(float(collateral_info['maxBorrowingAmount']),
                                    float(collateral_info['availableToBorrow']))
                if abs(qty) > borrow_amount:
                    msg += f'借貸上限為 {borrow_amount}，故修正顆數，'
                    qty = -borrow_amount

            # if the quantity is too small, set qty=0 to reject the order.
            qty = qty if abs(qty) >= float(
                self.target_spot_instrument_info['minOrderQty']) else 0

            usdt_amount = qty * self.api.get_spot_ask1_price(symbol)
            spot_maintain = usdt_amount*(1-float(collateral_info['collateralRatio']))
            contract_maintain = qty * self.api.get_contract_ask1_price(symbol)
            msg += f'受限於精細度及最小數量 {base_precision}，預計建倉 {qty} 顆 {self.target}，等值約 {usdt_amount:.2f} USDT。'
            self.update_action_text(msg)
            self.build_qty = qty
            self.open_good_side = ' (正優)' if qty > 0 else ' (負優)'
            # self.qty.set(qty)
            self.enable_arbitrage_on_click = True
            return True

        self.update_action_text('請在文字框中輸入符合格式之數字')
        return True

    def init_control_panel(self, frame):
        label = tk.Label(frame, text='請選擇貨幣')
        label.grid(row=0, column=0, padx=5)

        values = self.coin_list or [
            'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'DOT', 'MATIC',
            'LTC'
        ]
        values.insert(0, '')
        self.combobox = ttk.Combobox(frame, values=values, width=8)
        self.combobox.grid(row=0, column=1, padx=5)

        self.combobox.bind('<<ComboboxSelected>>', self.on_select_coin)

        vcmd = (frame.register(self.validation), '%P')
        vcmdu = (frame.register(partial(self.validation, is_usdt=True)), '%P')

        qty_label = tk.Label(frame, text='請輸入執行顆數')
        qty_label.grid(row=0, column=2, padx=5)

        self.qty = tk.StringVar()
        qty_entry = tk.Entry(frame,
                             textvariable=self.qty,
                             width=10,
                             validate='key',
                             validatecommand=vcmd)
        qty_entry.grid(row=0, column=3)

        usdt_amount_label = tk.Label(frame, text='或等值 USDT 金額')
        usdt_amount_label.grid(row=0, column=4, padx=5)

        self.usdt_amount = tk.StringVar()
        usdt_amount_entry = tk.Entry(frame,
                                     textvariable=self.usdt_amount,
                                     width=10,
                                     validate='key',
                                     validatecommand=vcmdu)
        usdt_amount_entry.grid(row=0, column=5)

        # 添加 Button 按鈕到 Frame
        button = tk.Button(frame, text='套利建倉', command=self.arbitrage_on_click)
        button.grid(row=0, column=6, padx=5)

        # 添加 Button 按鈕到 Frame
        cancel_button = tk.Button(frame, text='取消建倉', command=self.cancel_on_arbitrage)
        cancel_button.grid(row=0, column=7, padx=5)

        leverage_label = tk.Label(frame, text=f'槓桿倍數：{self.leverage} (目前固定)')
        leverage_label.grid(row=0, column=8)

        self.use_usdc_or_not = tk.BooleanVar()
        self.use_usdc = tk.Checkbutton(frame,
                                       text='是否使用 USDC 建倉',
                                       variable=self.use_usdc_or_not)

        self.use_usdc.grid(row=1, column=0, columnspan=2, sticky='nw')

        self.use_market_order_or_not = tk.BooleanVar()
        self.use_market_order = tk.Checkbutton(frame,
                                               text='是否市價建倉',
                                               variable=self.use_market_order_or_not)

        self.use_market_order.grid(row=1, column=2, columnspan=2, sticky='nw')

    def update_action_text(self, text):
        self.calculation_and_processing_canvas.itemconfig(self.action, text=text)

    def update_processing_text(self, text):
        self.calculation_and_processing_canvas.itemconfig(self.processing, text=text)

    def update_processing_bar_text(self, text):
        # work in progress, 預留版面

        # self.text_widget.insert(tk.END, text + '\n')
        # self.text_widget.see(tk.END)
        # text = f'[{get_time()}] {text}\n'
        # self.text_widget.insert('1.0', text)
        # self.text_widget.see('1.0')

        # self.log_fp.seek(0) # get to the first position
        # self.log_fp.write(text+'\n')
        pass

    def exit(self):
        self.app_is_killed = True
        for order_id, order in self.order_on_action.items():
            my_order = self.api.get_order_history(**order)
            if my_order['orderStatus'] not in ['Filled', 'Cancelled']:
                self.api.cancel_order(**order)
        # self.log_fp.close()
        self.app.destroy()

    @staticmethod
    def log_msg(*msgs):
        if DEBUG_MODE:
            msg = ' '.join([str(m) for m in msgs])
            print(f'[{get_time()}] {msg}')

    def cancel_on_arbitrage(self):
        self.continue_to_build = False

    def close_arbitrage_on_click(self):
        symbol = get_contract_symbol(self.coin_to_close)
        position_info = self.api.get_positions(symbol)
        position_size = position_size_with_sign(position_info.get('size', 0), position_info.get('side', ''))

        self.update_processing_text(f'準備進行平倉 {symbol}')
        self.log_msg(f'準備進行平倉 {symbol}')

        # spot size = - contract position
        # want to close -> build (- spot size) = contract position
        build_qty = position_size
        self.log_msg(f'欲建立反向倉位: {build_qty} 顆 {symbol} 現貨 = {-position_size} 顆 {symbol} 合約')
        args = dict(symbol=symbol,
                    contract_symbol=symbol,
                    build_qty=build_qty,
                    use_usdc=False,
                    use_market_order=self.use_market_order_or_not.get(),
                    )
        arbitrage_thread = threading.Thread(target=partial(self.arbitrage_building, **args), daemon=True)
        arbitrage_thread.start()

    def arbitrage_on_click(self):

        if not self.enable_arbitrage_on_click:
            self.update_processing_text('請確認建倉數量數字是否正確')
            self.log_msg('請確認建倉數量數字是否正確')
            return None

        self.update_processing_text('正在計算建倉數量...')
        self.log_msg('正在計算建倉數量...')

        use_usdc = False
        if self.use_usdc_or_not.get():
            symbol = get_contract_symbol(self.target, currency='USDC')
            usdc_symbol_spec = self.api.get_instruments_info('spot', symbol)
            use_usdc = (len(usdc_symbol_spec) > 0)
        if not use_usdc:
            symbol = get_contract_symbol(self.target)

        if not self.build_qty:  # 0 or None
            return None

        # set leverage
        usdt_symbol = get_contract_symbol(self.target)
        self.api.set_leverage(usdt_symbol, self.leverage)
        args = dict(symbol=symbol,
                    contract_symbol=usdt_symbol,
                    build_qty=self.build_qty,
                    use_usdc=use_usdc,
                    use_market_order=self.use_market_order_or_not.get()
                    )

        arbitrage_thread = threading.Thread(target=partial(self.arbitrage_building, **args), daemon=True)
        arbitrage_thread.start()

    def arbitrage_building(self,
                           symbol,
                           contract_symbol,
                           build_qty,
                           use_usdc=False,
                           use_market_order=False):
        """_summary_
            Open a position with the given quantity `linear_qty` with (limit order / market order)
            if use_market_order is (False / True).
            Then, in every 0.5 seconds, check that how many quantities are dealed
            and buy / sell the spot coin with the same amount.

        Args:
            symbol (_type_): _description_
            contract_symbol (_type_): _description_
            build_qty (_type_): _description_
            use_usdc (bool, optional): _description_. Defaults to False.
            use_market_order (bool, optional): _description_. Defaults to False.
        """
        linear_qty = - build_qty  # contract side is opposite to spot side

        self.log_msg(f'spot qty: {build_qty}, contract qty: {linear_qty}')

        coin = contract_symbol.split('USDT')[0]
        contract_side = 'Buy' if linear_qty > 0 else 'Sell'
        position_info = self.api.get_positions(contract_symbol)
        position_size = position_size_with_sign(position_info.get('size', 0), position_info.get('side', ''))
        target_position_size = position_size + linear_qty
        contract_ask1 = self.api.get_contract_ask1_price(contract_symbol)
        contract_bid1 = self.api.get_contract_bid1_price(contract_symbol)
        price = contract_bid1 if contract_side == 'Buy' else contract_ask1
        target_contract_instrument_info = self.api.get_instruments_info('linear', get_contract_symbol(coin))
        target_spot_instrument_info = self.api.get_instruments_info('spot', get_contract_symbol(coin))
        spot_base_precision = float(target_spot_instrument_info['basePrecision'])
        spot_min_order_qty = float(target_spot_instrument_info['minOrderQty'])

        self.continue_to_build = True

        place_order = self.api.place_order(
            category='linear',
            symbol=contract_symbol,
            side=contract_side,
            orderType='Limit' if not use_market_order else 'Market',
            qty=str(abs(linear_qty)),
            price=str(price),
        )
        order_id = place_order['result']['orderId']
        self.order_on_action[order_id] = dict(
            order_id=order_id, category='linear', symbol=contract_symbol)

        last_qty = None
        # to make the amount of spot = target_position_size
        while True:
            # check order status
            my_order = self.api.get_order_history(order_id=order_id,
                                                  category='linear',
                                                  symbol=contract_symbol)
            left_qty = float(my_order['leavesQty'])

            # if the order is calcelled, should update the target size to current position size
            if my_order['orderStatus'] == 'Cancelled':
                target_position_size = self.api.get_positions(
                    contract_symbol).get('size', 0)

            now_position_size = target_position_size - position_size_with_sign(left_qty, contract_side)

            spot_balance = self.api.get_spot_balance(coin=coin)
            qty_to_fill = (-now_position_size) - spot_balance

            # only buy side need to consider the fee
            if not use_usdc and qty_to_fill > 0:
                qty_to_fill = qty_to_fill / 0.999
            qty_to_fill = base_precision_filter(qty_to_fill, spot_base_precision, ceil=(qty_to_fill > 0))

            msgs = (f'訂單狀況: {my_order["orderStatus"]}, 掛單價格: {price}',
                    f'合約剩餘需建倉量: {position_size_with_sign(left_qty, contract_side)} 當下合約倉位: {now_position_size}',
                    f'建立至等值現貨倉位 {-now_position_size}, 需要買、賣的量: {qty_to_fill}',
                    )
            self.update_processing_text('\n'.join(msgs))
            if last_qty != left_qty:
                last_qty = left_qty
                for msg in msgs:
                    self.log_msg(msg)

            my_fill_order = None
            # 如果已有額外多的現貨, 但是合約還沒有建倉到位, 為了避免這邊因為對齊而賣出現貨, 要檢查 qty_to_fill 是否跟 linear_qty 方向相反
            if (abs(qty_to_fill) > spot_min_order_qty) and (qty_to_fill * linear_qty < 0):
                # 注意到市價單買入的 qty 會是使用 USDT 計量, 避免再換算, 所以這邊使用限價單, 未成交再交給下一輪迴圈處理
                spot_avg_price = self.api.get_price_to_fulfill_amount(
                    symbol=symbol, amount=qty_to_fill)
                new_fill_order = self.api.place_order(
                    category='spot',
                    symbol=symbol,
                    side='Buy' if qty_to_fill > 0 else 'Sell',
                    orderType='Limit',
                    qty=f'{abs(qty_to_fill)}',
                    price=f'{spot_avg_price}',
                )
                fill_order_id = new_fill_order['result']['orderId']
                my_fill_order = self.api.get_order_history(
                    order_id=fill_order_id, category='spot', symbol=symbol)

                act = '買' if qty_to_fill > 0 else '賣'
                msg = f'在均價 {spot_avg_price} {act} {abs(qty_to_fill)} 顆 {symbol.split("USDT")[0]}'
                self.update_processing_bar_text(msg)
                self.log_msg(msg)

            if left_qty == 0 and (my_fill_order is None
                                  or my_fill_order['orderStatus']
                                  in ['Filled', 'Cancelled']):
                break
            else:
                # 限價單若未完全成交，則取消剩下，等下一輪限價單重新成交
                if my_fill_order is not None and (my_fill_order['orderStatus'] not in ['Filled', 'Cancelled']):
                    self.api.cancel_order(category='spot',
                                          symbol=contract_symbol,
                                          order_id=fill_order_id)
                    fill_order_id = None

            if not self.continue_to_build:
                if my_order['orderStatus'] not in ['Cancelled', 'Filled']:
                    self.api.cancel_order(category='linear',
                                          symbol=contract_symbol,
                                          order_id=order_id)
                    my_order['orderStatus'] = 'Cancelled'
                break

            time.sleep(0.5)

        self.continue_to_build = False

        if my_order['orderStatus'] == 'Cancelled':
            self.update_action_text(f'訂單已取消。')
            self.update_processing_text('')
            self.log_msg(f'訂單已取消。')
        else:
            self.update_action_text(f'建/平倉完畢。')
            self.update_processing_text('')
            self.log_msg(f'建/平倉完畢。')
        self.order_on_action.pop(order_id)
        self.update_wallet()

    def on_select_coin(self, event):
        coin = event.widget.get()
        self.qty.set('')
        self.usdt_amount.set('')
        if coin == '':
            self.target = None
            self.build_qty = None
            self.display_price_canvas = False
            return

        self.set_target(coin)
        if not self.display_price_canvas:
            self.display_price_canvas = True

    def set_target(self, coin):
        self.target = coin
        self.target_contract_instrument_info = self.api.get_instruments_info(
            'linear', get_contract_symbol(coin))
        self.target_spot_instrument_info = self.api.get_instruments_info(
            'spot', get_contract_symbol(coin))
        self.set_precision()

    def set_precision(self):
        tick_size = min(float(self.target_contract_instrument_info['tickSize']), float(
            self.target_spot_instrument_info['tickSize']))
        self.precision = get_precision(tick_size)

    def balance_spots_with_positions(self):
        balance_thread = threading.Thread(target=self._balance_spots_with_positions, daemon=True).start()

    def _balance_spots_with_positions(self):
        my_coins = self.api.get_wallet_balance()

        for coin, equity in my_coins.items():
            symbol = get_contract_symbol(coin)
            if symbol in STABLE_COINS:
                continue
            position_info = self.api.get_positions(symbol)
            spot_tickers = self.api.get_spot_tickers(symbol)
            contract_tickers = self.api.get_contract_tickers(symbol)
            equity = to_float(equity)

            position = position_size_with_sign(
                position_info['size'], position_info['side'])
            spot_to_balance = -(equity + position)
            # only buy side need to consider the fee
            spot_intrument_info = self.api.get_instruments_info('spot', symbol)
            spot_min_qty = float(spot_intrument_info['minOrderQty'])
            spot_base_precision = float(spot_intrument_info['basePrecision'])
            spot_to_balance = base_precision_filter(spot_to_balance, spot_base_precision, ceil=(spot_to_balance > 0))

            if abs(spot_to_balance) >= spot_min_qty:
                spot_price = spot_tickers['ask1Price'] if spot_to_balance > 0 else spot_tickers['bid1Price']
                place_order = self.api.place_order(
                    category='spot',
                    symbol=symbol,
                    side='Buy' if spot_to_balance > 0 else 'Sell',
                    orderType='Limit',
                    qty=str(abs(spot_to_balance)),
                    price=str(spot_price),
                )
                action = '買入' if spot_to_balance > 0 else '賣出'
                self.log_msg(f'{action} 現貨 {spot_to_balance} 使期現貨平衡。')
            else:
                self.log_msg(f'現貨 {symbol} 量不足最小下單數量。')

        self.update_wallet()

    def on_select_coin_to_close(self, event):
        self.coin_to_close = event.widget.get()

    def run_update_price_canvas(self):
        update_price_thread = threading.Thread(target=self._run_update_price_canvas, daemon=True)
        update_price_thread.start()

    def _run_update_price_canvas(self):
        while not self.app_is_killed:
            time.sleep(0.5)
            if self.target is None:
                self.target_price_canvas.itemconfig(self.price_diff_info, text='')
                continue
            symbol = get_contract_symbol(self.target)

            available_balance = self.api.get_available_balance()
            try:
                usdc_usd_indexprice = self.api.get_spot_usd_index_price('USDCUSDT')
                self.usdc_usd_indexprice = usdc_usd_indexprice
            except:
                pass
            usdc_usdt_price = self.api.get_spot_ask1_price('USDCUSDT')
            usdt_usd_indexprice = self.usdc_usd_indexprice / usdc_usdt_price

            spot_usdt_bid1 = self.api.get_spot_bid1_price(symbol)
            spot_usdt_ask1 = self.api.get_spot_ask1_price(symbol)
            contract_ask1 = self.api.get_contract_ask1_price(symbol)
            contract_bid1 = self.api.get_contract_bid1_price(symbol)
            build_price_diff = contract_ask1 - spot_usdt_ask1
            close_price_diff = contract_ask1 - spot_usdt_bid1
            funding_rate = self.api.get_funding_rate(symbol)
            fr = float(funding_rate['fundingRate'])
            next_funding_time = miniseconds_to_datetime(
                int(funding_rate['nextFundingTime']) - get_time_stamp())
            collateral_info = self.api.get_collateral_info(self.target)
            borrowable = collateral_info['borrowable']
            hourly_borrow_rate = collateral_info['hourlyBorrowRate']
            br = '不允許借貸' if not borrowable else f'{8*float(hourly_borrow_rate)*100: .4f}% (8hr)'
            price_format = f'>{self.digit}.{self.precision}f'

            qty_step = max(float(self.target_contract_instrument_info['qtyStep']), float(
                self.target_spot_instrument_info['basePrecision']))
            qty_precision = get_precision(qty_step)
            qty_format = f'>{self.digit}.{qty_precision}f'
            msg = ''

            # 利用公式計算出來的數量跟網站上顯示的還是不一樣, 不確定還有什麼參數沒有用到....
            # 所以保險起見, 開倉數量先都除以 2.

            # 正向資費
            # available_balance > initial_margin = (contract_bid1 * qty * (1/self.leverage + 0.055*0.01*2) * usdt_usd_indexprice +
            #                                       spot_usdt_ask1 * qty * (1+0.1*0.01*2) * usdt_usd_indexprice)
            # 負向資費
            # available_balance > initial_margin = (contract_ask1 * qty + (1/self.leverage + 0.00055*2) * usdt_usd_indexprice +
            #                                       spot_usdt_bid1 * qty * (
            #                                           max( 1/10, (1+1/spot_leverage)/float(collateral_info['collateralRatio'])-1 )
            #                                           + 0.001 * 2
            #                                       ) * usdt_usd_indexprice
            max_positive_building_qty = available_balance / \
                ((contract_bid1 * (1/self.leverage + 0.00055*2) + spot_usdt_ask1 * (1+0.001*2)) * usdt_usd_indexprice)
            max_negative_building_qty = None
            if borrowable:
                self.api.set_spot_margin_leverage(leverage=2)
                spot_leverage = 2
                max_negative_building_qty = available_balance / ((contract_ask1 * (1/self.leverage + 0.00055*2) + (
                    max(1/10, (1+1/spot_leverage)/float(collateral_info['collateralRatio'])-1) * spot_usdt_bid1)) * usdt_usd_indexprice)

            msg += (f'目前可用餘額 {available_balance:>16.8f} USD\n\n'
                    + f'正向資費套利，建議安全可建倉量為 {max_positive_building_qty/2:{qty_format}} 顆。\n'
                    + f'負向資費套利，建議安全可建倉量為 {-max_negative_building_qty/2:{qty_format}} 顆。\n' if max_negative_building_qty is not None else ''
                    )

            msg += '\n'
            msg += (f'貨幣對 {self.target}/USDT ［現貨賣一價］ {spot_usdt_ask1:{price_format}} ［合約賣一價］ {contract_ask1:{price_format}} '
                    + f'［資金費率］{fr*100: .4f}% ({next_funding_time}) \n'
                    + f'{" "*(11+len(self.target))} ［現貨買一價］ {spot_usdt_bid1:{price_format}} ［合約買一價］ {contract_bid1:{price_format}} '
                    + f'［借貸利率］{br}\n\n'
                    + f'［USDT 建倉價差{self.open_good_side}］{(build_price_diff)/(spot_usdt_ask1+1e-10)*100: 5.4f}%\n')

            # 確認是否有 coin/USDC 商品
            symbol = get_contract_symbol(self.target, currency='USDC')
            usdc_symbol_spec = self.api.get_instruments_info('spot', symbol)
            if len(usdc_symbol_spec):
                spot_usdc_bid1 = self.api.get_spot_bid1_price(symbol)
                spot_usdc_ask1 = self.api.get_spot_ask1_price(symbol)
                usdc_ask1 = self.api.get_spot_ask1_price('USDCUSDT')
                usdc_bid1 = self.api.get_spot_bid1_price('USDCUSDT')
                build_price_diff = contract_ask1 - spot_usdc_ask1 * usdc_ask1
                close_price_diff = spot_usdc_bid1 * usdc_bid1 - contract_ask1
                msg += f'［USDC 建倉價差{self.open_good_side}］{build_price_diff/(spot_usdc_ask1 * usdc_ask1 + 1e-10)*100: 5.4f}%'

            else:
                msg += f'［USDC 建倉價差{self.open_good_side}］無 USDC 交易對'

            self.target_price_canvas.itemconfig(self.price_diff_info, text=msg)

    def init_wallet(self, frame):
        self.canvas = tk.Canvas(frame,
                                width=self.width,
                                height=self.wallet_display_frame_height)
        self.canvas.pack(padx=0, pady=0)

        # 顯示錢包資訊
        self.wallet_tree = ttk.Treeview(self.canvas,
                                        columns=tuple(WALLET_COLUMNS.keys()),
                                        show='headings')
        self.wallet_tree.tag_configure('odd',
                                       background='#005050',
                                       foreground='white',
                                       font=('Courier', 13),
                                       )
        self.wallet_tree.tag_configure('even',
                                       background='#303050',
                                       foreground='white',
                                       font=('Courier', 13),
                                       )

        for col, text in WALLET_COLUMNS.items():
            self.wallet_tree.heading(col, text=text)
            self.wallet_tree.column(col, width=COLUMNS_WIDTH[col], anchor='nw')
        self.wallet_tree.pack(padx=0, pady=0)

        self.update_wallet()

    def update_wallet(self):
        self.update_margin_rate()
        # 清除現有數據
        for row in self.wallet_tree.get_children():
            self.wallet_tree.delete(row)

        my_coins = self.api.get_wallet_balance()

        tag = False
        for coin in STABLE_COINS:
            if coin not in my_coins:
                continue
            tag = not tag
            self.wallet_tree.insert('',
                                    'end',
                                    values=(coin, my_coins[coin], '', '', '', '', '', '', '', ''),
                                    tags=('odd' if tag else 'even', ))

        # 可以先查詢倉位, 有現貨的一起列, 沒有現貨的另外列出來
        # https://bybit-exchange.github.io/docs/zh-TW/v5/position

        for coin, equity in my_coins.items():
            symbol = get_contract_symbol(coin)
            if symbol in STABLE_COINS:
                continue
            position_info = self.api.get_positions(symbol)
            position_size = position_size_with_sign(position_info['size'], position_info['side'])

            # TODO?
            # 是否能偵測有任何合約倉位就顯示? <- 應該可以

            # if position_size == 0:
            #     continue

            # 再思考一下怎麼表現開倉與平倉價差的優劣方向

            # TODO2

            # 最大可建倉數的計算怎麼做? 維持保證金率的公式有點難計算....

            tag = not tag
            funding_rate = self.api.get_funding_rate(symbol)['fundingRate']
            spot_tickers = self.api.get_spot_tickers(symbol)
            contract_tickers = self.api.get_contract_tickers(symbol)
            equity = to_float(equity)

            spot_open_price, spot_close_price = to_float(
                spot_tickers['ask1Price']), to_float(spot_tickers['bid1Price'])
            contract_open_price, contract_close_price = to_float(
                contract_tickers['bid1Price']), to_float(spot_tickers['ask1Price'])

            # 負向資費, 開平倉方向相反, 最優價格互換
            # 因此若平倉價差是負優
            avg_8hr_borrow_rate = ''
            if equity < 0:
                spot_open_price, spot_close_price = spot_close_price, spot_open_price
                contract_open_price, contract_close_price = contract_close_price, contract_open_price

                # borrow_history = self.api.get_borrow_history_logs(coin)
                # if len(borrow_history):
                #     avg_8hr_borrow_rate = borrow_history['hourlyBorrowRate'].iloc[:8].apply(lambda x: float(x)).mean()*8
                avg_8hr_borrow_rate = self.api.get_borrow_rate(coin)*8

            open_price_diff = contract_open_price - spot_open_price
            close_price_diff = contract_close_price - spot_close_price

            equity_usdt_value = equity * spot_close_price

            if position_size != 0:
                self.my_positions.append(coin)
            try:
                contract_logs = self.api.get_transaction_logs(
                    category='linear',
                    coin=coin,
                    quantity=position_size,
                )
            except Exception as e:
                self.log_msg(position_info)
                raise Exception('Error in get_transaction_logs', e)

            spot_logs = self.api.get_transaction_logs(
                category='spot',
                coin=coin,
                quantity=to_float(my_coins[coin]),
            )

            filter = (spot_logs['currency'] == 'USDT') | (spot_logs['currency']
                                                          == 'USDC')
            to_usdt_prices = spot_logs[filter]['usdcPrice'].apply(
                lambda x: 1 if np.isnan(x) else to_float(x, 1))
            spot_cost = -(
                spot_logs[filter].change.apply(lambda x: to_float(x, 0)) *
                to_usdt_prices).sum()
            spot_pnl = equity_usdt_value - spot_cost

            realized_pnl = contract_logs.change.apply(
                lambda x: to_float(x, 0)).sum()
            unrealized_pnl = to_float(position_info['unrealisedPnl'], 0)

            self.wallet_tree.insert(
                '',
                'end',
                values=(symbol, equity, f'{spot_pnl: .4f}',
                        f'{float(position_size): .4f}',
                        f'{float(funding_rate)*100: 7.4f}%',
                        f'{avg_8hr_borrow_rate*100: 7.4f}%' if avg_8hr_borrow_rate != '' else '',
                        f'{realized_pnl: .4f}',
                        unrealized_pnl,
                        f'{spot_pnl+realized_pnl+unrealized_pnl: .4f}',
                        spot_close_price, contract_close_price,
                        f'{close_price_diff/spot_close_price*100: .3f}%',
                        ),
                tags=('odd' if tag else 'even', ))


if __name__ == '__main__':

    executor = ArbitrageExecutor()
    executor.app.protocol('WM_DELETE_WINDOW', executor.exit)

    try:
        executor.app.mainloop()
    except KeyboardInterrupt:
        executor.exit()
