from common.types import Venue

def get_collector_functions(venue: Venue):
    if venue == venue.BINANCE:
        from inputs.collector_binance import fetch_klines, health_check
        return fetch_klines, health_check
    elif venue == Venue.YAHOO:
        raise NotImplementedError(f"Collector functions not implemented for this venue: {venue}")
    elif venue == venue.MT5:
        from inputs.collector_mt5 import fetch_klines, health_check
        return fetch_klines, health_check
    else:
        raise ValueError(f"Unknown collector type: {venue}")

def get_download_functions(venue: Venue):
    if venue == venue.BINANCE:
        from inputs.collector_binance import download_klines
        return download_klines
    elif venue == Venue.YAHOO:
        from inputs.collector_yahoo import download_klines
        return download_klines
    elif venue == venue.MT5:
        from inputs.collector_mt5 import download_klines
        return download_klines
    else:
        raise ValueError(f"Unknown venue {venue} or downloader for the venue not implemented")
