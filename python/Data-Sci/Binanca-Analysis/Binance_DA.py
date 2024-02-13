from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd

## Authentication
apiKey = ''
secretKey = ''
client = Client(apiKey, secretKey)

## Get Tickers
tickers = client.get_all_tickers()
# print(tickers[1])

ticker_df = pd.DataFrame(tickers)
print(ticker_df.head())
ticker_df.set_index('symbol', inplace=True)
# print(float(ticker_df.loc['BTCUSDT']['price']))

## Get Depth
depth = client.get_order_book(symbol='ETHBTC')
# print(depth)

depth_df = pd.DataFrame(depth['asks'])
depth_df.columns = ['Price', 'Volume']
# print(depth_df.head())
# print(depth_df.dtypes())

## Get Historical Data
historical = client.get_historical_klines('ETHBTC', Client.KLINE_INTERVAL_1DAY, '1 Jan 2011')
hist_df = pd.DataFrame(historical)
hist_df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                    'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

## Preprocess Historical Data
hist_df['Open Time'] = pd.to_datetime(hist_df['Open Time']/1000, units='s')
hist_df['Close Time'] = pd.to_datetime(hist_df['Close Time']/1000, units='s')

numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'TB Base Volume', 'TB Quote Volume']
hist_df[numeric_columns] = hist_df[numeric_columns].apply(pd.to_numeric, axis=1)

# print(hist_df.describe())
# print(hist_df.info())

## Visualisation
import mplfinance as mpf

mpf.plot(hist_df.set_index('Close Time').tail(120),
         type='candle',
         style='charles',
         volume=True,
         title='ETHBTC Last 120 Days',
         mav=(10,20,30))