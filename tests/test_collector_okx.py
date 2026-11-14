"""Tests fuer inputs/collector_okx.py - ohne Netz (fetch wird eingespeist)."""
import pandas as pd

from inputs import collector_okx as ok

H = 3_600_000


def kerze(ts, o=1.0, bestaetigt="1"):
    # OKX: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    return [str(ts), str(o), str(o + 1), str(o - 1), str(o + 0.5), "10", "10", "12", bestaetigt]


def test_freq_umrechnung():
    assert ok.okx_bar("1h") == "1H"
    assert ok.okx_bar("1min") == "1m"
    assert ok.okx_bar("4h") == "4H"
    assert ok.okx_bar("1d") == "1D"


def test_inst_id_aus_ordnername():
    assert ok.inst_id("BTCUSDT") == "BTC-USDT"
    assert ok.inst_id("ETHUSDC") == "ETH-USDC"
    assert ok.inst_id("BTC-USDT-SWAP") == "BTC-USDT-SWAP"


def test_df_hat_binance_spalten_und_aufsteigende_zeit():
    zeilen = [kerze(3 * H), kerze(2 * H), kerze(1 * H)]          # OKX liefert absteigend
    df = ok.rows_to_df(zeilen, H)
    assert list(df.columns) == ok.column_names
    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["close_time"].iloc[0] == 1 * H + H - 1
    assert df["quote_av"].iloc[0] == 12.0


def test_unbestaetigte_kerze_wird_verworfen():
    df = ok.rows_to_df([kerze(2 * H, bestaetigt="0"), kerze(1 * H)], H)
    assert len(df) == 1


def test_paginierung_bis_max_rows_und_ohne_doppelte():
    seiten = {
        None: [kerze(t * H) for t in range(10, 7, -1)],            # 10, 9, 8
        str(8 * H): [kerze(t * H) for t in range(7, 4, -1)],       # 7, 6, 5
        str(5 * H): [kerze(t * H) for t in range(4, 1, -1)],       # 4, 3, 2
    }
    aufrufe = []

    def fetch(inst, bar, after):
        aufrufe.append(after)
        return seiten.get(after, [])

    df = ok.hole_kerzen("BTC-USDT", "1H", H, max_rows=6, fetch=fetch, pause=0)
    assert len(df) == 6
    assert df["timestamp"].iloc[-1] == pd.Timestamp(10 * H, unit="ms", tz="UTC")
    assert len(aufrufe) == 2                                       # stoppt, sobald genug da ist


def test_paginierung_endet_bei_leerer_seite():
    df = ok.hole_kerzen("BTC-USDT", "1H", H, max_rows=0,
                        fetch=lambda i, b, a: [] if a else [kerze(5 * H)], pause=0)
    assert len(df) == 1
