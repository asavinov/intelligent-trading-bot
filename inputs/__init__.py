from common.types import Venue


def get_collector_functions(venue: Venue):
    if venue == venue.BINANCE:
        from inputs.collector_binance import sync_data_collector_task, data_provider_health_check
        return sync_data_collector_task, data_provider_health_check
    elif venue == venue.MT5:
        from inputs.collector_mt5 import sync_data_collector_task, data_provider_health_check
        return sync_data_collector_task, data_provider_health_check
    else:
        raise ValueError(f"Unknown collector type: {venue}")
