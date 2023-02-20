import click

from backtesting import Backtest, Strategy

from service.App import *
from service.analyzer import *

import logging
log = logging.getLogger('backtest')

@click.command()
@click.option('--config_file', '-c', type=click.Path(), default='', help='Configuration file name')
def main(config_file):
    load_config(config_file)
    symbol = App.config["symbol"]
    time_column = App.config["time_column"]
    log.info(f"Starting backtesting for: {symbol}. ")
    
    App.analyzer = Analyzer(App.config)
    data_path = Path(App.config["data_folder"]) / symbol
    file_path = (data_path / App.config.get("feature_file_name")).with_suffix(".csv")
    if not file_path.is_file():
        log.error(f"Data file does not exist: {file_path}")
        return
    
    log.info(f"Loading data from source data file {file_path}...")
    df = pd.read_csv(file_path, parse_dates=[time_column], index_col=time_column)

    df = App.analyzer.generate_score(df)
    df = App.analyzer.aggregate_post_process(df)

    signal_model = App.config['signal_model']
    apply_rule_with_score_thresholds(df, signal_model, 'buy_score_column', 'sell_score_column')
    if len(df.loc[df["buy_signal_column"] | df["sell_signal_column"]]) == 0:
        log.info("No buy or sell signals in dataset")
        return

    df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
    log.info(f"Finished loading {len(df)} records with {len(df.columns)} columns.")

    class ITB(Strategy):
        def init(self):
            pass
        def next(self):
            buy_signal = self.data.df["buy_signal_column"].iloc[-1:].values[0]
            sell_signal = self.data.df["sell_signal_column"].iloc[-1:].values[0]

            if buy_signal == True:
                if self.position.is_short:
                    self.position.close()
                elif not self.position.is_long:
                    self.buy(size=.1)
            if sell_signal == True:
                if self.position.is_long:
                    self.position.close()
                elif not self.position.is_short:
                    self.buy(size=.1)

    bt = Backtest(df, ITB, cash=50000, commission=.002)
    stats = bt.run()
    bt.plot()
    log.info(stats)

if __name__ == '__main__':
    main()
