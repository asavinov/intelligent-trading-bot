import MetaTrader5 as mt5

class AccountBalances:
    """
    Available assets for trade
    """
    # Can be set by the sync/recover function or updated by the trading algorithm
    base_quantity = "0.04108219"  # BTC owned (on account, already bought, available for trade)
    quote_quantity = "1000.0"  # USDT owned (on account, available for trade)


class MT5AccountInfo(mt5.AccountInfo):
    pass
