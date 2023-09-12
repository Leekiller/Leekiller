import backtrader as bt
import backtrader.indicators as ta

class TValue(bt.Indicator):
    """

    Preparing for TSV idicator
    """
    params = (('tsv_length', 13), ('tsv_ma_length', 7))
    lines = ('t_value',)

    def __init__(self):
        return None

    def next(self):
        self.l.t_value[0] = (self.data.volume[0] 
                             * (self.data.close[0]-self.data.close[-1])
                             + self.p.tsv_length)

class TSV(bt.Indicator):
    """

    Time Segmented Volume
    """
    params = (('tsv_length', 13), ('tsv_ma_length', 7))
    lines = ('tsv',)

    def __init__(self):
        self.t_value = TValue(tsv_length=self.p.tsv_length)
        self.l.tsv = ta.SMA(self.t_value, period=self.p.tsv_ma_length)
