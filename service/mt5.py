from typing import Optional
import MetaTrader5 as mt5
import logging

log = logging.getLogger('mt5')

def connect_mt5(mt5_account_id: Optional[int] = None, mt5_password: Optional[str] = None, mt5_server: Optional[str] = None, **kwargs):
    """
    Initializes the MetaTrader 5 connection and attempts to log in with the provided credentials.
    """
    # Initialize MetaTrader 5 connection
    if not mt5.initialize():
        log.error(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    log.info(f"MT5 Initialized. Version: {mt5.version()}")

    if mt5_account_id and mt5_password and mt5_server:
        authorized = mt5.login(int(mt5_account_id), password=str(mt5_password), server=str(mt5_server), **kwargs)
        if not authorized:
            log.error(f"MT5 Login failed for account #{mt5_account_id}, error code: {mt5.last_error()}")
            mt5.shutdown()
            return False
    return True
