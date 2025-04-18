from common.types import Venue


def get_collector_functions(venue: Venue):
    if venue == venue.BINANCE:
        from collector_binance import main_collector_task, data_provider_health_check, sync_data_collector_task
        return main_collector_task, data_provider_health_check, sync_data_collector_task
    elif venue == venue.MT5:
        from collector_mt5 import main_collector_task, data_provider_health_check, sync_data_collector_task
        return main_collector_task, data_provider_health_check, sync_data_collector_task
    else:
        raise ValueError(f"Unknown collector type: {venue}")
