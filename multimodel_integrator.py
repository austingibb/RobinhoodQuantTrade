from model_simulator import ModelSim
from model_sim_constants import ActionResult

DEBUG_PRINT = True

def default_buy_func(expected_price):
    return ActionResult.SUCCESS, expected_price

def default_sell_func(expected_price):
    return ActionResult.SUCCESS, expected_price

class ModelIntegrator:
    """
    Takes in multiple models and simulates them all in parallel, using a metric to decide which one to listen
    to for any given trade action.
    """
    def __init__(self, model_func_dictionary,
                 asset_amount_per_action,
                 buy_func=default_buy_func,
                 sell_func=default_sell_func,
                 historical_data=[], starting_cash=100.0):
        self.model_sim_dictionary = {}
        for key, model_func in model_func_dictionary.items():
            self.model_sim_dictionary[key] = ModelSim(model_func, historical_data,
                                                      starting_cash=starting_cash, asset_amount_per_action=asset_amount_per_action)
        self.all_asset_data_points = historical_data.copy()

        self.asset_amount_per_action = asset_amount_per_action
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.asset_owned = False

        self.buy_func = buy_func
        self.sell_func = sell_func

        self.current_model = list(model_func_dictionary.keys())[0]
        self.net_asset_history = []
        self.continual_profit_history = []
        self.transactional_profit_history = []
        self.progress_start_index = len(historical_data)
        self.buy_indices = []
        self.sell_indices = []
        self.previous_buy_amount = 0
        self.transactional_profit = 0

    def progress_models(self, asset_price):
        """
        Takes in a datapoint representing the most recent price, and calls the model with the information.
        Also handles deciding between the multiple inputs from the models.
        :param asset_price: Most recent price.
        :return: The latest model decision.
        """

        current_model_decision = 0
        self.all_asset_data_points.append(asset_price)
        # Get starting index ignoring the initial training indices
        progress_index = len(self.all_asset_data_points) - self.progress_start_index - 1

        # A given model is in control, until another one has the highest metric
        if self.asset_owned:
            for key, model_sim in self.model_sim_dictionary.items():
                decision, metric = model_sim.progress_model(asset_price)
                if key == self.current_model:
                    current_model_decision = decision

        else:
            highest_metric_value = None
            highest_metric_model = None
            highest_metric_decision = None
            for key, model_sim in self.model_sim_dictionary.items():
                decision, metric = model_sim.progress_model(asset_price)
                if highest_metric_value == None or metric > highest_metric_value:
                    highest_metric_value = metric
                    highest_metric_model = key
                    highest_metric_decision = decision

            self.current_model = highest_metric_model
            current_model_decision = highest_metric_decision

        if DEBUG_PRINT:
            print("Current model: ", self.current_model, " Current model decision: ", current_model_decision)

        # buy
        if current_model_decision == -1:
            if self.asset_owned == False:
                result, buy_amount = self.buy_func(asset_price * self.asset_amount_per_action)
                assert result == ActionResult.SUCCESS, "Integrator buy failed"
                assert buy_amount > 0, "Buy amount must be positive"

                self.previous_buy_amount = buy_amount
                self.cash -= buy_amount
                self.asset_owned = True

                self.buy_indices.append(progress_index)
            else:
                current_model_decision = 0

        #sell
        elif current_model_decision == 1:
            if self.asset_owned == True:
                result, sell_amount = self.sell_func(asset_price * self.asset_amount_per_action)
                assert result == ActionResult.SUCCESS, "Sell failed"
                assert sell_amount > 0, "Sell amount must be positive"

                self.transactional_profit += sell_amount - self.previous_buy_amount
                self.cash += sell_amount
                self.asset_owned = False

                self.sell_indices.append(progress_index)
            else:
                current_model_decision = 0

        self.net_asset_history.append(self.get_net_assets())
        self.continual_profit_history.append(self.get_continual_profit())
        self.transactional_profit_history.append(self.get_transactional_profit())

        return current_model_decision

    def get_continual_profit(self):
        return self.get_net_assets() - self.starting_cash

    def get_transactional_profit(self):
        return self.transactional_profit

    def get_net_assets(self):
        # asset value + cash amount = total assets
        net_assets = self.get_cash()
        if self.asset_owned:
            net_assets += self.get_asset_value()
        return net_assets

    def get_cash(self):
        return self.cash

    def get_asset_value(self):
        if self.asset_owned:
            # not necessarily exact to live data, it just uses the most recent data point to calculate the value of all assets
            return self.asset_amount_per_action * self.all_asset_data_points[-1]
        else:
            return 0

    def get_continual_profit_history(self):
        return self.continual_profit_history

    def get_transactional_profit_history(self):
        return self.transactional_profit_history

    def get_asset_history(self):
        if len(self.all_asset_data_points) > self.progress_start_index:
            return self.all_asset_data_points[self.progress_start_index:].copy()

    def get_transaction_indices(self):
        return self.buy_indices, self.sell_indices

    def get_current_model(self):
        return self.get_model(self.current_model)

    def get_model(self, model_key):
        return self.model_sim_dictionary[model_key]

    def has_asset(self):
        return self.asset_owned

    def get_current_model_metric(self):
        return self.get_model_metric(self.current_model)

    def get_model_plot_data(self, model_key):
        return self.model_sim_dictionary[model_key].get_graph_data_points()

    def get_model_metric(self, model_key):
        return self.model_sim_dictionary[model_key].get_metric()