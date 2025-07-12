import os
import ccxt
import pandas as pd
import pandas_ta as pta
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

proxy_url = "socks5://127.0.0.1:10810"


class TrendPairLists:
    
    def __init__(self, exchange = 'binance', enable_proxy = False, data_dir='user_data/data'):
        exchange_class = getattr(ccxt, exchange)
        self.exchange = exchange_class()
        self.exchange.enableRateLimit = True
        
        if enable_proxy:
            self.exchange.socksProxy = proxy_url
            
        self.timeframes = ['30m', '1d']
        self.futures_symbols = []
        
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
    def run(self):
        top = 100
        self.futures_symbols = self.get_top_volume_futures_symbols(top=top)
        # self.fetch_futures_data(self.futures_symbols, limit=150)
        all_data = self.load_futures_data(self.futures_symbols)
        self.calc_indicators(all_data)
        
    def get_top_volume_futures_symbols(self, top):
        try:
            markets = self.exchange.load_markets()
            futures_symbols = []
            for symbol, market in markets.items():
                if market.get('contract') == True and market.get('expiry') is None and market.get('expiryDatetime') is None and symbol.endswith(':USDT'):
                    futures_symbols.append(symbol)

            tickers = self.exchange.fetch_tickers(futures_symbols)
            futures_volumes = []
            for symbol, ticker in tickers.items():
                if symbol in futures_symbols:
                    ticker = tickers.get(symbol)
                    futures_volumes.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume']
                    })
                    
            sorted_futures = sorted(futures_volumes, key=lambda x: x['volume'], reverse=True)[:top]
            futures_symbols = [f['symbol'] for f in sorted_futures]
            
            logger.info(f"Fetched {len(futures_symbols)} top volume futures symbols from {self.exchange.id}")
            return futures_symbols
        except Exception as e:
            logger.error(f"Fetch futures symbols error: {e}")
            return []
    
    def fetch_futures_data(self, futures_symbols, limit=100, delay=0.1):
        if not futures_symbols:
            return {}
        
        all_data = {}
        total_requests = len(futures_symbols) * len(self.timeframes)
        current_request = 0
        
        for symbol in futures_symbols:
            all_data[symbol] = {}
            
            for timeframe in self.timeframes:
                current_request += 1
                logger.info(f"Fetching {symbol} {timeframe} candles: {current_request/total_requests:.2%}({current_request}/{total_requests})")
                
                df = self.fetch_ohlcv(symbol, timeframe, limit)
                
                if df is not None and not df.empty:
                    all_data[symbol][timeframe] = df
                    symbol_name = self.format_symbol(symbol)
                    df.to_csv(os.path.join(self.data_dir, f"{symbol_name}_{timeframe}.csv"), index=False)
                    logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
                else:
                    logger.warning(f"No candles data for {symbol} {timeframe}")
                
                # time.sleep(delay)
        
        return all_data
    
    def load_futures_data(self, futures_symbols):
        all_data = {}
        
        for symbol in futures_symbols:
            all_data[symbol] = {}
            
            for timeframe in self.timeframes:
                symbol_name = self.format_symbol(symbol)
                file_path = os.path.join(self.data_dir, f"{symbol_name}_{timeframe}.csv")
                if os.path.exists(file_path):
                    logger.info(f"Loading {symbol} {timeframe} candles from {file_path}")
                    df = pd.read_csv(file_path)
                    all_data[symbol][timeframe] = df
                    
        return all_data
    
    def calc_indicators(self, all_data):
        for symbol, timeframes in all_data.items():
            for timeframe, df in timeframes.items():
                if df.empty:
                    continue
                
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                df['rsi'] = self.calculate_rsi(df['close'], period=14)
                logger.info(f"Calculated indicators for {symbol} {timeframe}")
        
    
    def format_symbol(self, symbol):
        return symbol.replace('/', '-').replace(':', '_')

    def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe, 
                since=since, 
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            return df
        except Exception as e:
            logger.error(f"Fetched {symbol} {timeframe} candles error: {e}")
            return None


if __name__ == "__main__":
    trend_pair_lists = TrendPairLists(exchange = 'binance', enable_proxy = True)
    trend_pair_lists.run()