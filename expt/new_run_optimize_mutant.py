from leekiller.optimizer import DE

class Optimizer(DE):

    def __init__(self):
        self.dataframe = None
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

    def load_data(data_path: str=None):
        if data_path is None:
            data_path = "../data/BTCUSD_latest.csv"
        dataframe = pd.read_csv(data_path, parse_dates=True, index_col=0)
        dataframe.index = pd.to_datetime(dataframe.index, format='ISO8601')
        self.dataframe = dataframe
        return None

    def get_objective_value(self, control_params: dict):
        pass


def main():
    """ To do

    Move create_populations into __init__()
    """
    optimizer = Optimizer()
    optimizer.create_populations()
    optimizer.run(itr=30)
    pass

if __name__=="__main__":
    main()
