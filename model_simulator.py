from model_sim_constants import ActionResult

def default_buy_func(expected_price):
    return ActionResult.SUCCESS, expected_price

def default_sell_func(expected_price):
    return ActionResult.SUCCESS, expected_price

class ModelSim:
    """
    Class to facilitate the simulation of a model, separate from the real trading decision.
    """
    def __init__(self, model_func,
                 historical_data=[],
                 buy_func=default_buy_func,
                 sell_func=default_sell_func,
                 starting_cash=5000.0,
                 asset_amount_per_action=0.016):
        """
        Initialize the model to begin processing data points.

        :param model_func: Function to produce a buy/sell/hold decision.
        :param historical_data: Array of data points to process.
        :param buy_func: Function to perform purchase action.
        :param sell_func: Function to perform sell action.
        :param starting_cash: Amount of "fake cash" to start the model with.
        :param asset_amount_per_action: What percentage of the assets should be moved on each trading action.
        """

        # passed amounts
        self.all_asset_data_points = historical_data.copy()
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.asset_amount_per_action = asset_amount_per_action

        # modular functions
        self.buy_func = buy_func
        self.sell_func = sell_func
        self.model_func = model_func

        # calculation/initiation
        self.asset_owned = False
        self.graph_data_points = []
        self.net_asset_history = []
        self.most_recent_metric = 0

    def progress_model(self, asset_price):
        self.all_asset_data_points.append(asset_price)
        # model is called here, decision is the buy/sell/hold decision, metric is to measurement of performance, and plotdata is for graphing
        (original_decision, metric, plot_data) = self.model_func(self.all_asset_data_points)
        self.most_recent_metric = metric
        # print(decision)
        assert (original_decision > -2 and original_decision < 2), "Model must always return 1, 0, or -1"

        print(self.all_asset_data_points)
        print(original_decision)
        self.graph_data_points = plot_data.copy()

        # If the model provides a buy decision multiple times it effectively does nothing.
        # buy
        if original_decision == -1:
            # if we don't already have an asset
            if not self.asset_owned:
                result, buy_amount = self.buy_func(asset_price * self.asset_amount_per_action)
                assert result == ActionResult.SUCCESS, "Buy failed"
                assert buy_amount > 0, "Buy amount must be positive"

                self.cash -= buy_amount
                self.asset_owned = True
        # sell
        elif original_decision == 1:
            # if we actually have something to sell
            if self.asset_owned:
                result, sell_amount = self.sell_func(asset_price * self.asset_amount_per_action)
                assert result == ActionResult.SUCCESS, "Sell failed"
                assert sell_amount > 0, "Sell amount must be positive"

                self.cash += sell_amount
                self.asset_owned = False

        # hold results in no action

        self.net_asset_history.append(self.get_net_assets())

        return original_decision, metric

    def get_net_assets(self):
        """
        Provides total asset value, including cash and asset value.
        :return:
        """
        net_assets = self.get_cash()
        if self.asset_owned:
            net_assets += self.get_asset_value()
        return net_assets

    def get_profit(self):
        return self.get_net_assets() - self.starting_cash

    def get_cash(self):
        return self.cash

    def get_asset_value(self):
        """
        Get's the assets value as of the last trade.
        :return: Asset value as of the last trade.
        """
        if self.asset_owned:
            # not necessarily exact to live data, it just uses the most recent data point to calculate the value of all assets
            return self.asset_amount_per_action * self.all_asset_data_points[-1]
        else:
            return 0

    def get_graph_data_points(self):
        return self.graph_data_points

    def get_net_asset_history(self):
        return self.net_asset_history

    def has_asset(self):
        return self.asset_owned

    def get_metric(self):
        return self.most_recent_metric