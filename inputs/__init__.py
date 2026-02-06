from common.types import Venue

def get_collector_functions(venue: Venue):
    if venue == venue.BINANCE:
        from inputs.collector_binance import fetch_klines, health_check
        return fetch_klines, health_check
    elif venue == venue.MT5:
        from inputs.collector_mt5 import fetch_klines, health_check
        return fetch_klines, health_check
    else:
        raise ValueError(f"Unknown collector type: {venue}")

def get_download_functions(venue: Venue):
    if venue == venue.BINANCE:
        from inputs.download_binance import download_binance
        return download_binance
    elif venue == Venue.YAHOO:
        from inputs.download_yahoo import download_yahoo
        return download_yahoo
    elif venue == venue.MT5:
        from inputs.download_mt5 import download_mt5
        return download_mt5
    else:
        raise ValueError(f"Unknown venue {venue} or downloader for the venue not implemented")
