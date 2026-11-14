"""Historische Kerzen von OKX (oeffentlich, kein Schluessel) im Binance-Spaltenformat.

Die CSV hat dieselben Spalten wie collector_binance, damit Features, Labels und
Training unveraendert weiterlaufen. Nur Download; ein Live-Collector (fetch_klines)
gibt es hier bewusst nicht - dieser Zweig ist der lesende Teil, Handel laeuft nicht
ueber dieses Modul.

OKX-Endpunkt: GET /api/v5/market/history-candles?instId=&bar=&after=&limit=100
Antwort absteigend: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm].
"""
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASIS = "https://www.okx.com"            # OKX_BASE (z. B. EEA-Host) ueberschreibt
HEADERS = {"User-Agent": "Mozilla/5.0"}   # ohne User-Agent antwortet OKX mit HTTP 403
SEITE = 100
PAUSE_S = 0.12          # history-candles: 20 Anfragen je 2 s

column_names = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
]
column_types = {
    'timestamp': 'datetime64[ns, UTC]',
    'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64',
    'close_time': 'int64',
    'quote_av': 'float64', 'trades': 'int64', 'tb_base_av': 'float64', 'tb_quote_av': 'float64',
    'ignore': 'float64',
}
_BAR = {"1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W"}
_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
       "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000, "6H": 21_600_000,
       "12H": 43_200_000, "1D": 86_400_000, "1W": 604_800_000}


def okx_bar(freq):
    """Pandas-Frequenz ("1h", "1min") -> OKX-Balkenlaenge ("1H", "1m")."""
    try:
        return _BAR[str(freq).lower()]
    except KeyError:
        raise ValueError("Frequenz nicht abbildbar auf OKX: %r" % (freq,))


def inst_id(name):
    """Ordnername wie BTCUSDT -> BTC-USDT; Namen mit '-' bleiben."""
    if "-" in name:
        return name
    for quote in ("USDT", "USDC", "USD", "EUR", "BTC", "ETH"):
        if name.endswith(quote) and len(name) > len(quote):
            return name[:-len(quote)] + "-" + quote
    raise ValueError("Instrument nicht ableitbar: %r" % (name,))


def rows_to_df(rows, bar_ms):
    """OKX-Zeilen -> DataFrame im Binance-Format, aufsteigend, nur bestaetigte Kerzen."""
    daten = []
    for r in rows:
        if len(r) >= 9 and str(r[8]) != "1":
            continue
        ts = int(r[0])
        daten.append([ts, r[1], r[2], r[3], r[4], r[5], ts + bar_ms - 1,
                      r[7] if len(r) > 7 else 0, 0, 0, 0, 0])
    df = pd.DataFrame(daten, columns=column_names)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.astype(column_types)
    return df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)


def _fetch_http(inst, bar, after):
    q = {"instId": inst, "bar": bar, "limit": str(SEITE)}
    if after:
        q["after"] = after
    basis = (os.environ.get("OKX_BASE") or BASIS).rstrip("/")
    url = basis + "/api/v5/market/history-candles?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=15) as r:
        d = json.load(r)
    if d.get("code") != "0":
        raise OSError("OKX %s: %s" % (d.get("code"), d.get("msg")))
    return d["data"]


def hole_kerzen(inst, bar, bar_ms, max_rows=0, fetch=_fetch_http, pause=PAUSE_S, bis_ts=None):
    """Seitenweise rueckwaerts holen, bis max_rows erreicht, bis_ts unterschritten oder Ende."""
    rows, after = [], None
    while True:
        seite = fetch(inst, bar, after)
        if not seite:
            break
        rows.extend(seite)
        aeltester = int(seite[-1][0])
        if max_rows and len(rows) >= max_rows:
            break
        if bis_ts is not None and aeltester <= bis_ts:
            break
        after = str(aeltester)
        if pause:
            time.sleep(pause)
    df = rows_to_df(rows, bar_ms)
    return df.tail(max_rows) if max_rows else df


def download_klines(config, data_sources):
    """Wie collector_binance.download_klines: schreibt <data_folder>/<folder>/klines.csv."""
    time_column = config["time_column"]
    data_path = Path(config["data_folder"])
    download_max_rows = config.get("download_max_rows", 0)
    bar = okx_bar(config["freq"])
    bar_ms = _MS[bar]
    for ds in data_sources:
        quote = ds.get("folder")
        if not quote:
            print("ERROR. Folder is not specified.")
            continue
        inst = ds.get("inst_id") or inst_id(quote)
        print("Start downloading '%s' (%s, %s) from OKX ..." % (quote, inst, bar))
        ordner = data_path / quote
        ordner.mkdir(parents=True, exist_ok=True)
        datei = (ordner / "klines").with_suffix(".csv")
        df_alt, bis = None, None
        if datei.is_file():
            df_alt = pd.read_csv(datei)
            df_alt[time_column] = pd.to_datetime(df_alt[time_column], format='ISO8601', utc=True)
            df_alt = df_alt.astype(column_types)
            bis = int(df_alt[time_column].iloc[-5:].iloc[0].timestamp() * 1000)   # kleine Ueberlappung
        df = hole_kerzen(inst, bar, bar_ms, max_rows=0 if bis else download_max_rows, bis_ts=bis)
        if df_alt is not None:
            df = pd.concat([df_alt, df]).drop_duplicates(subset=[time_column], keep="last")
            df = df.sort_values(time_column).reset_index(drop=True)
        if download_max_rows:
            df = df.tail(download_max_rows)
        df.to_csv(datei, index=False)
        print("Finished downloading '%s'. Stored %d rows in '%s'" % (quote, len(df), datei))
