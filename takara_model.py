import numpy as np
import statistics as s
import math
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', np.RankWarning)

def takara_model_calc(stock, param):

    window = param[0]
    trend_window = param[1]

    buy_plot_x = []
    buy_plot_y = []
    sell_plot_x = []
    sell_plot_y = []


    buy_vec_y = np.array([])
    sell_vec_y = np.array([])

    avg_plot = []
    upper_plot = []
    lower_plot = []
    stock_plot = []

    check = 0
    profit = 0
    decision = 0
    metric = 0

    for n in range(window, len(stock) + 1):

        stock_feed = np.array(stock[0:n])
        time_feed = np.arange(0, n)


        # CALCULATIONS###############################################################################
        if trend_window == 0:
            slope = 0
        else:
            slope = np.polyfit(time_feed[n - trend_window:n], stock_feed[n - trend_window:n], 1)[0]

        avg = s.mean(stock_feed[n - window:n])
        std = s.stdev(stock_feed[n - window:n])

        upper_lim = avg + 3 * std / math.sqrt(window)
        lower_lim = avg - 3 * std / math.sqrt(window)

        # BUY SELL RULES#############################################################################
        if (stock_feed[-1] <= lower_lim) and slope >= -.1 and check == 0:
            decision = -1  # buy
            check = 1

            buy_plot_y.append(stock_feed[-1])
            buy_plot_x.append(time_feed[-1]-window+2)

            buy_vec_y = np.append(buy_vec_y, stock_feed[-1])

        elif stock_feed[-1] >= upper_lim and check == 1:
            decision = 1  # sell
            check = 0

            sell_plot_y.append(stock_feed[-1])
            sell_plot_x.append(time_feed[-1] - window+2)

            sell_vec_y = np.append(sell_vec_y, stock_feed[-1])

            profit = np.sum(sell_vec_y - buy_vec_y)

            metric = get_ratio(buy_vec_y,sell_vec_y)
        else:
            decision = 0  # do nothing



    ############################################################################

        stock_plot.append(stock_feed[-1])
        avg_plot.append(avg)
        upper_plot.append(upper_lim)
        lower_plot.append(lower_lim)

    x_val = []
    for q in range(1, len(stock_plot)+1):
        x_val.append(q)

    #plt.plot(x_val, stock_plot, 'b')
    #plt.plot(x_val, avg_plot, 'k')
    #plt.plot(x_val, upper_plot, 'r')
    #plt.plot(x_val, lower_plot, 'r')
    #plt.plot(buy_plot_x, buy_plot_y,'ro')
    #plt.plot(sell_plot_x, sell_plot_y,'go')
    #plt.draw()
    #plt.pause(2)
   # plt.close()

    plot_info = [[x_val, stock_plot, 'b'], [x_val, avg_plot, 'y'], [x_val, upper_plot, 'm'], [x_val, lower_plot, 'm'], [buy_plot_x, buy_plot_y, 'ro'], [sell_plot_x, sell_plot_y, 'go']]
   # plot_info = []


    output = [decision, metric, plot_info]
    return output

# Structure Search Method to find optimal parameters for model
def takara_model(stock_total):

    stock = stock_total[-300:]

    # Divide to train and test
    training_val = .6
    num_train = int(round(len(stock) * training_val))

    training_stock = stock[:num_train]
    testing_stock = stock[num_train:]

    max_ratio = 0
    max_param = []

    for n in range(4, 9):
        for m in range(10, 14):
            param = [n, m]

            temp = takara_model_calc(training_stock, param)[1]

            if temp > max_ratio:
                max_param = param
                max_ratio = temp
    #print(max_param)
    [decision, metric, plot_info] = takara_model_calc(testing_stock, max_param)

    if metric >= 10000:
        metric = metric/10000

    if metric<1 and decision != 1:
        decision = 0

    return [decision, metric, plot_info]


def get_ratio(buy_vec, sell_vec):

    if len(buy_vec) > len(sell_vec):
        del buy_vec[-1]
    if len(sell_vec) > len(buy_vec):
        del sell_vec[-1]

    gain = 0
    loss = 0

    for m in range(0, len(buy_vec)):
        temp = sell_vec[m] - buy_vec[m]

        if temp > 0:
            gain = gain + temp
        else:
            loss = loss + abs(temp)

    if gain > 0 and loss > 0:
        ratio = gain / loss
    elif gain > 0 and loss == 0:
        ratio = gain * 10000
    else:
        ratio = 0

    return ratio
