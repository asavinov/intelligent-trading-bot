from common.types import Venue


def get_trader_functions(venue: Venue) -> dict[str, callable]:
    """
    Return a dict of the four trader-related callables for the given venue.

    Example:
        funcs = get_trader_functions(Venue.BINANCE)
        funcs["trader"](...)
        funcs["update_order_status"](...)
    """
    if venue == venue.BINANCE:
        from outputs.trader_binance import (
            trader_binance,
            update_account_balance as update_account_balance_binance,
            update_order_status as update_order_status_binance,
            update_trade_status as update_trade_status_binance,
        )
        return {
            "trader": trader_binance,
            "update_account_balance": update_account_balance_binance,
            "update_order_status": update_order_status_binance,
            "update_trade_status": update_trade_status_binance,
        }
    elif venue == venue.MT5:
        from outputs.trader_mt5 import (
            trader_mt5,
            update_account_balance as update_account_balance_mt5,
            update_order_status as update_order_status_mt5,
            update_trade_status as update_trade_status_mt5,
        )

        return {
            "trader": trader_mt5,
            "update_account_balance": update_account_balance_mt5,
            "update_order_status": update_order_status_mt5,
            "update_trade_status": update_trade_status_mt5,
        }
    else:
        raise ValueError(f"Unknown trader venue: {venue!r}")
