import numpy as np
import statistics as s
import random as r
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import pickle

def Sort_Data( Raw_data , Data_per_sample , Training_ratio ):
    features = []
    targets = []
    Price_table = []
    for i in range( Data_per_sample , len( Raw_data) ):
        temp = Raw_data[ i - Data_per_sample : i ]
        feature = temp
        Price_table.append( temp[-1] )
        # Possible:
        #feature.append( s.mean(temp) )
        #feature.append( s.stdev(temp) )
        #feature.append( np.polyfit( list(range(len(temp))) , temp , 1 )[0] )

        features.append( feature )
        # Target is the future stock price
        targets.append( Raw_data[i] )

    Last_data_point = Raw_data[-1*Data_per_sample:]
    
    # Reshape
    num_train = int(round( len(features )*Training_ratio ))
    features = np.array( features )

    # Divide to train and test
    Training_features = features[:num_train]
    Training_targets = targets[:num_train]
    Testing_features = features[num_train:]
    Testing_targets = targets[num_train:]
    Price_table = Price_table[num_train:]
     
    # Scale the features
##    scaler = MinMaxScaler()
##    scaler.fit( features )
##    Training_features  = scaler.transform( Training_features )
##    Testing_features = scaler.transform( Testing_features )
##    Last_data_point = scaler.transform( np.array(Last_data_point).reshape(1,-1) )
    Last_data_point =  np.array(Last_data_point).reshape(1,-1) 
    # Shuffle the training set
    temp = list(zip( Training_features , Training_targets ))
    r.shuffle( temp )
    Training_features[:] , Training_targets[:] = zip(*temp)

    return [ Training_features , Training_targets , Testing_features , Testing_targets , Price_table , Last_data_point]



# MLR model, optimized using max profit on test set
def MLR_model ( Training_features , Training_targets , Testing_features , Testing_targets , stock_limit , Price_table ):
    max_profit_to_loss = -10000000000000000
    min_error = 1
    for L1 in [ 40 , 80  ]:
        for L2 in [ 20  ]:
            for L3 in [ 10  ]:
                for alpha in [0.001 ]:
                    MLPR = MLPRegressor( activation = 'relu' , hidden_layer_sizes = (L1, L2 , L3 ) , solver = 'lbfgs' , alpha = alpha , batch_size = 50 , learning_rate = 'adaptive' , max_iter = 2000000 ,early_stopping=True)
                    MLPR.fit( Training_features , Training_targets )
                    Train_predict = MLPR.predict( Training_features )
                    predict_price = []
                    for q in range( len(Testing_features) ):
                        predict_price.append(MLPR.predict( Testing_features[q].reshape(1, -1) ))
                        
                    error = np.mean( abs(np.array(Train_predict )-  np.array(Training_targets))  / np.array(Training_targets) )
                    # Calculate profit
                    stock = 0
                    loss = 0
                    brought = 0
                    profit = 0
                    # For plotting
                    t = list(range(len(Testing_targets)))
                    BP = []
                    B = []
                    SP = []
                    S = [] 
                    for i in range( 6, len( predict_price) ):
                        if ( i % 1 == 0 ):
                            current_price = Price_table[i]
                            if (  stock <=  predict_price[i] and stock > 0 ): # Sell the stock if you have it as soon as hiting a down trend
                                if ( (current_price - stock) >= 0 ):
                                    profit += (current_price - stock)
                                else:
                                    loss += abs( current_price - stock )
                                stock = 0
                                brought = 0
                                SP.append( t[i] )
                                S.append( current_price )              
                            elif ( current_price <=  predict_price[i]  and stock == 0): # Buy stock at the bottom of the down trend
                                stock = current_price
                                brought = 1
                                BP.append( t[i] )
                                B.append( current_price )

                    if ( loss == 0 ):
                        loss = 0.0001
                    if ( profit/loss >= max_profit_to_loss ):
                        model = MLPR
                        max_profit = profit
                        max_profit_to_loss = profit/loss
                        prices = predict_price
                        Buy_points = BP
                        Buys = B
                        Sell_points = SP
                        Sells = S
    # Save optimal model
    joblib.dump(model, 'MLPR.pkl')
    plot_data = [ [t , Price_table , 'm' ] , [np.array(t) , prices , 'b'] , [Buy_points , Buys , 'ro'] , [Sell_points , Sells , 'go'] ]
    metric = max_profit_to_loss
    if ( metric >= 10000 ):
        metric /= 15000


    return [ max_profit  , metric ,plot_data ]
    
def jimmy_model(Raw_data):
    import pickle
    # How many data point in a sample
    Data_per_sample = 5
    # How much percentage of the total data goes to training
    Training_ratio = 0.6
    # How many stock we can have max
    stock_limit = 30
    try:
        F = open('stock.pkl' , 'rb')
        stock = pickle.load(F)
        F.close()
    except:
        stock = 0
    [ Training_features , Training_targets , Testing_features , Testing_targets , Price_table , Last_data_point] = Sort_Data( Raw_data , Data_per_sample , Training_ratio )
    [ max_profit , metric , plot_data ] = MLR_model ( Training_features , Training_targets , Testing_features , Testing_targets , stock_limit , Price_table )
    Model = joblib.load('MLPR.pkl')
    latest_predict = Model.predict( Last_data_point )
    current_price = Raw_data[-1]
    last_predict = Model.predict( np.array(Testing_features[-1]).reshape(1,-1))
    if ( latest_predict >= stock and stock > 0):
        decision = 1 # Sell
        F = open('stock.pkl' , 'wb' )
        pickle.dump( 0 , F )
        F.close()
    elif ( latest_predict < current_price ):
        decision = -1 # Buy
        F = open('stock.pkl' , 'wb' )
        pickle.dump( current_price , F )
        F.close()
    else:
        decision = 0

    if metric<1 and decision != 1:
        decision = 0
    # metric = metric/10

    return [ decision , metric , plot_data ]





