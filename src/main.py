import pandas as pd 

from conv_model import ConvCryptoTimeSeriesModel
from q_seq_model import QSequenceCryptoTimeSeriesModel
# from lstm_model import LSTMCryptoTimeSeriesModel

if __name__ == "__main__":
    # Define standard variables
    datafile = "ETH-USD.csv"
    index_col =  "Date"
    y_col = 0
    writefile = "feb24.csv"
    num_train = 100
    num_test = 50
    batch_size = 10
    start_index = 200
    lookback = 5
    lr = .005
    y_test = None
    y_preds = {}

    # Define models
    model1 = ConvCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        epochs = 1000,
        lr = lr,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
        conv = [128, 64], 
        quantum=False
    )
    model2 = ConvCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        epochs = 1000,
        lr = lr,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
        conv = [128, 64], 
        quantum=True
    )
    model3 = QSequenceCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        epochs = 1000,
        lr = lr,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
        num_layers = 2,
    )
    # Models to try out 
    models = [model1]

    # Run each model
    for i in range(len(models)):
        model = models[i]
        model.read(datafile,index_col)
        model.preprocess(y_col = y_col)
        if type(model) == ConvCryptoTimeSeriesModel: model.initialize_layers()
        model.train()
        model.test()
        model.write(writefile=writefile)
        y_test, y_pred_i = model.invtransform_y()
        if i > 0: y_preds[model.type] = y_pred_i
        plotfile = "{}_class_{}_ex_{}.png".format(model.type, model.y_col, 3)
        model.plot(plotfile=plotfile)

    # Plot final results
    # model1.plot(plotfile="Multiple_class_{}_ex_{}".format(model1.y_col, 3), y_preds=y_preds)


