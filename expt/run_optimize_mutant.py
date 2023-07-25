import datetime
import sys
import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.indicators as ta
from mutant.model import Mutant
from mutant.strategy import MutantBacktrader
from leekiller.optimizer import DE

raw_data_path = "../data/BTCUSD_latest.csv"
init_protfolio_value = 100000.0

# def _load_data():
dataframe = pd.read_csv(raw_data_path,
                        parse_dates=True,
                        index_col=0)
dataframe.index = pd.to_datetime(dataframe.index, format='ISO8601')

def objective(control_params):
    control_params['tp'] = [0.6]
    control_params['sl'] = [1.8]
    model = Mutant(**control_params)
    control_params.pop('tp')
    control_params.pop('sl')
    trade_reports = []
    # drawdown_reports = []
    roi = []
    total_sessions = 3
    for i in range(total_sessions):
        backtest_length = 1440*30
        start = np.random.choice(len(dataframe))
        while start > (len(dataframe) - backtest_length):
            start = np.random.choice(len(dataframe))
        end = start + backtest_length
        df = dataframe.iloc[start:end]
        start_date = dataframe.iloc[start].name
        end_date = dataframe.iloc[end].name
        df = df.groupby(pd.Grouper(freq='5Min')).agg({"open": "first", 
                                                      "high": "max",
                                                      "low": "min",
                                                      "close": "last",
                                                      "volume": "sum"})
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None)
        cerebro = bt.Cerebro()
        cerebro.addstrategy(MutantBacktrader, model, print_log=False)
        cerebro.adddata(data)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='mutant_trade')
        # cerebro.addanalyzer(bt.analyzers.DrawDown, _name='mutant_drawdown')
        cerebro.addsizer(bt.sizers.PercentSizer, percents=10)
        cerebro.broker.setcash(init_protfolio_value)
        cerebro.broker.setcommission(commission=0.0004)
        # print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        results = cerebro.run()
        result = results[0]
        trade_reports.append(result.analyzers.mutant_trade.get_analysis())
        # drawdown_reports.append(result.analyzers.mutant_drawdown.get_analysis())
        # print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        current_report = trade_reports[-1]
        key = 'pnl'
        if current_report['total']['total'] > 0 and key in current_report.keys():
            pnl = current_report['pnl']['net']['total']
        else:
            print("Start: %s, End: %s" %(start_date, end_date))
            print("model.params: ")
            print(model.params)
            print("current_report: ")
            print(current_report)
            pnl = 0
        roi.append(pnl/init_protfolio_value * 100)
    roi_avg = sum(roi) / len(roi)
    return roi_avg

# def main():
model = Mutant()
control_params = model.params
control_params
control_params.pop('tp')
control_params.pop('sl')
control_params_range = {
    'ema_1_length': [5, 300],
    'ema_2_length': [5, 300],
    'ema_3_length': [5, 300],
    'macd_fast_length': [5, 300],
    'macd_slow_length': [5, 300],
    'macd_signal_length': [5, 300],
    'macd_average_length': [5, 300],
    'rsi_length': [5, 300],
    'rsi_long': [10, 90],
    'rsi_short': [10, 90]
}

# Load optimizer
optimizer = DE(objective, control_params, control_params_range)
optimizer.create_populations()
optimizer.run(itr=300)

# if __name__=="__main__":
#     main()