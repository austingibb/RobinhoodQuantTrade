from robinhood_crypto_api import RobinhoodCrypto

DEBUG_PRINT = True

class BtcData:
    def __init__(self, r):
        self.r = r
        self.current_data = []

        historical_info = self.r.historicals(interval='15second', span='hour')
        data_points = historical_info['data_points']

        data_index = 0
        for d in data_points:
            self.current_data.append(float(d['open_price']))
            data_index += 1

    def pull_new_data(self):
        market_quote = float(self.r.quotes()['mark_price'])
        self.current_data.append(market_quote)
        self.current_data.pop(0)
        return market_quote

    def get_current_data(self):
        return self.current_data

class BuySellBtc:
    def __init__(self, crypto_session, btc_amount):
        self.crypto_session = crypto_session
        self.btc_amount = btc_amount

    def buy(self):
        quote_info = self.crypto_session.quotes()
        market_order_info = self.crypto_session.trade(
            'BTCUSD',
            price=round(float(quote_info['mark_price']) * 1.003, 2),
            quantity=str(self.btc_amount),
            side="buy",
            time_in_force="ioc",
            type="market"
        )

        if DEBUG_PRINT:
            print(market_order_info)

    def sell(self):
        quote_info = self.crypto_session.quotes()
        market_order_info = self.crypto_session.trade(
            'BTCUSD',
            price=round(float(quote_info['mark_price']) * 0.997, 2),
            quantity=str(self.btc_amount),
            side="sell",
            time_in_force="ioc",
            type="market"
        )
        if DEBUG_PRINT:
            print(market_order_info)