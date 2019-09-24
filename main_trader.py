from robinhood_crypto_api import RobinhoodCrypto
from model_simulator import ModelSim
from model_sim_constants import ActionResult
from multimodel_integrator import ModelIntegrator
from takara_model import takara_model
from jimmy_model import jimmy_model
from robinhood_bitcoin_wrapper import BtcData, BuySellBtc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import random, csv, time

REALTIME = 0
HISTORICAL_DEMO = 1
RUN_TYPE = REALTIME

def get_historical_points(r):
    historical_info = r.historicals()
    data_points = historical_info['data_points']

    open_prices_fiveminutes = []
    for d in data_points:
        open_prices_fiveminutes.append(float(d['open_price']))

    return open_prices_fiveminutes

def main():
    bitcoin_size = 0.00013

    if RUN_TYPE == REALTIME:
        r = RobinhoodCrypto("austingibb@gmail.com", "AkV92HoZ*1ir")

        btc_data = BtcData(r)
        bs_btc = BuySellBtc(r, bitcoin_size)

    def buy_btc_func(expected_amount):
        bs_btc.buy()
        return ActionResult.SUCCESS, expected_amount

    def sell_btc_func(expected_amount):
        bs_btc.sell()
        return ActionResult.SUCCESS, expected_amount

    def buy_dummy_func(expected_amount):
        return ActionResult.SUCCESS, expected_amount

    def sell_dummy_func(expected_amount):
        return ActionResult.SUCCESS, expected_amount

    buy_func = buy_btc_func if RUN_TYPE == REALTIME else buy_dummy_func
    sell_func = sell_btc_func if RUN_TYPE == REALTIME else sell_dummy_func
    run_model_integrator(buy_func, sell_func, btc_data, bitcoin_size=bitcoin_size, graph_decisions=True)

def run_model_integrator(buy_func, sell_func, bitcoin_size, data_source, run_type=HISTORICAL_DEMO, graph_decisions=True):
    model_dict = {}
    model_dict['model_one'] = takara_model
    model_dict['model_two'] = jimmy_model

    # model_integrator = ModelIntegrator(model_dict, 0.001, historical_data=btctousd_interval[0:1000])
    model_integrator = ModelIntegrator(model_dict, bitcoin_size, buy_func=buy_func, sell_func=sell_func, historical_data=data_source.get_current_data())
    while True:
    # for btctousd_data_point_index in range(1000, len(btctousd_interval)):
        time.sleep(15)
        price = data_source.pull_new_data()
        # btctousd_data_point = btctousd_interval[btctousd_data_point_index]
        decision = model_integrator.progress_models(price)
        # decision = model_integrator.progress_models(btctousd_data_point)

        print("Price: ", price, " Current winning metric: ", model_integrator.get_current_model_metric())
        print("#", 0, "Net: $", model_integrator.get_net_assets(), " Cash: $", model_integrator.get_cash(),
              " Asset Val: $", model_integrator.get_asset_value(), " Has Asset: ", model_integrator.has_asset())

        takara_model_data = model_integrator.get_model('model_one')
        jimmy_model_data = model_integrator.get_model('model_two')

        print("model_one #", 0, "Net: $", takara_model_data.get_net_assets(), " Cash: $", takara_model_data.get_cash(),
              " Asset Val: $", takara_model_data.get_asset_value(), " Metric: ", takara_model_data.get_metric(), " Has Asset: ", takara_model_data.has_asset())
        print("model_two #", 0, "Net: $", jimmy_model_data.get_net_assets(), " Cash: $", jimmy_model_data.get_cash(),
              " Asset Val: $", jimmy_model_data.get_asset_value(), " Metric: ", jimmy_model_data.get_metric(), " Has Asset: ", jimmy_model_data.has_asset())
        print()

        if decision == -1:
            print("Bought")
            print()
        elif decision == 1:
            print("Sold")
            print()

        buy_indices, sell_indices = model_integrator.get_transaction_indices()
        if graph_decisions:
            graph_all(model_integrator.get_model_metric('model_one'), model_integrator.get_model_metric('model_two'),
                    model_integrator.get_transactional_profit(), model_integrator.get_model_plot_data('model_one'), model_integrator.get_model_plot_data('model_two'),
                    model_integrator.get_asset_history(), model_integrator.get_transactional_profit_history(), buy_indices, sell_indices)

def graph_all(metric_1, metric_2, profit, model_one_plot_data, model_two_plot_data,
              historic_asset_data, net_profit_data, buy_indices, sell_indices):
    """
    Rather hacky function for drawing trade data. This is a hackathon after all.
    """
    gs = gridspec.GridSpec(8, 8)
    ############################################################################################
    m1 = metric_1
    m2 = metric_2
    net_profit = str(profit)

    ax1 = plt.subplot(gs[1:4, 0:3], facecolor='k')
    ax2 = plt.subplot(gs[5:8, 0:3], facecolor='k')
    ax3 = plt.subplot(gs[1:4, 4:], facecolor='k')
    ax4 = plt.subplot(gs[5:8, 4:], facecolor='k')

    for i in range(len(model_one_plot_data)):
        x = model_one_plot_data[i][0]
        y = model_one_plot_data[i][1]
        s = model_one_plot_data[i][2]
        ax1.plot(x, y, s)

    for i in range(len(model_two_plot_data)):
        x = model_two_plot_data[i][0]
        y = model_two_plot_data[i][1]
        s = model_two_plot_data[i][2]
        ax2.plot(x, y, s)

    x_values_historic = list(range(0, len(historic_asset_data)))
    x_values_profit = list(range(0, len(net_profit_data)))
    ax3.plot(x_values_historic, historic_asset_data)


    for buy_index in buy_indices:
        ax3.plot(buy_index, historic_asset_data[buy_index], 'ro')

    for sell_index in sell_indices:
        ax3.plot(sell_index, historic_asset_data[sell_index], 'go')

    ax4.plot(x_values_profit, net_profit_data)
    #############################################################################################
    ax1.set_title("Algorithmic Trading Bot\n\n\nModel 1")
    ax1.set_ylabel('Price [$]')
    ax1.set_xlabel('Time [ticks]')
    ax1.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        right=False,
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off

    ax2.set_title("Model 2")
    ax2.set_ylabel('Price [$]')
    ax2.set_xlabel('Time [ticks]')
    ax2.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        right=False,
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off

    ax3.set_title("Final Buy/Sell Decision ")
    ax3.set_ylabel('Price [$]')
    ax3.set_xlabel('Time [ticks]')
    ax3.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        right=False,
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off

    ax4.set_title("Net Profit")
    ax4.set_ylabel('Price [$]')
    ax4.set_xlabel('Time [ticks]')
    ax4.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        left=False,
        right=False,
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off

    ax1.text(0.99, 0.01, m1,
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax1.transAxes,
             color='green', fontsize=10)

    ax2.text(0.99, 0.01, m2,
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax2.transAxes,
             color='green', fontsize=10)

    label_profit = 'Profit: '
    ax4.text(0.99, 0.01, label_profit + net_profit,
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax4.transAxes,
             color='green', fontsize=10)

    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':
    main()



