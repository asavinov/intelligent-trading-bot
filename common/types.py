from decimal import Decimal
from enum import Enum
import MetaTrader5 as mt5

class Venue(Enum):
    BINANCE = "binance"
    MT5 = "mt5"
    
class AccountBalances:
    """
    Available assets for trade
    """
    # Can be set by the sync/recover function or updated by the trading algorithm
    base_quantity = "0.04108219"  # BTC owned (on account, already bought, available for trade)
    quote_quantity = "1000.0"  # USDT owned (on account, available for trade)


# mt5.AccountInfo
class MT5AccountInfo:
    balance: Decimal = "10000"
    equity: Decimal = "10000"
    margin: Decimal = "0"
    margin_free: Decimal = "10000"
    margin_level: Decimal = "10000"
    profit: Decimal = "0"
    login: int = 0
    currency: str = "USD"
    name: str = ""
    server: str = ""
    leverage: int = 1


class MT5OrderStatus(Enum):
    NEW = mt5.ORDER_STATE_PLACED
    PARTIALLY_FILLED = mt5.ORDER_STATE_PARTIAL
    FILLED = mt5.ORDER_STATE_FILLED
    CANCELED = mt5.ORDER_STATE_CANCELED
    PENDING_CANCEL = mt5.ORDER_STATE_REQUEST_CANCEL
    REJECTED = mt5.ORDER_STATE_REJECTED
    EXPIRED = mt5.ORDER_STATE_EXPIRED
    
