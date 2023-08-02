import numpy as np
import pandas as pd
import backtrader as bt
from mutant.model import Mutant
from mutant.strategy import MutantBacktrader
from leekiller.optimizer import DE

class Optimizer(DE):

    def __init__(self):
        self.model = Mutant()
        self.control_params = self.model.params
        self.control_params_range = {
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
        super().__init__()

    def load_data(self, data_path: str=None):
        if data_path is None:
            data_path = "../data/BTCUSD_latest.csv"
        dataframe = pd.read_csv(data_path, parse_dates=True, index_col=0)
        dataframe.index = pd.to_datetime(dataframe.index, format='ISO8601')
        self.dataframe = dataframe
        return None

    def get_objective_value(self, control_params: dict) -> float:
        """ Avraged ROI as objective value

        """
        self.model.update_params(control_params)
        """Test print"""
        trade_reports = []
        roi = []
        session_win = 0
        total_sessions = 10
        init_protfolio_value = 100000.0
        for i in range(total_sessions):
            backtest_length = 1440*30
            start = np.random.choice(len(self.dataframe))
            while start > (len(self.dataframe) - backtest_length):
                start = np.random.choice(len(self.dataframe))
            end = start + backtest_length
            df = self.dataframe.iloc[start:end]
            df = df.groupby(pd.Grouper(freq='5Min')).agg({"open": "first", 
                                                          "high": "max",
                                                          "low": "min",
                                                          "close": "last",
                                                          "volume": "sum"})
            data = bt.feeds.PandasData(dataname=df,datetime=None)
            cerebro = bt.Cerebro()
            cerebro.addstrategy(MutantBacktrader, self.model, print_log=False)
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
                if pnl > 0:
                    session_win += 1
            else:
                pnl = 0
            # roi.append(pnl/init_protfolio_value * 100)
        # roi_avg = sum(roi) / len(roi)
        session_win_rate = session_win/total_sessions
        return session_win_rate


def main():
    optimizer = Optimizer()
    optimizer.load_data()
    optimizer.run(itr=10, batch=4)
    pass

if __name__=="__main__":
    main()
