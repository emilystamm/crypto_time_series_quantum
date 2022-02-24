import pandas as pd 

from conv_model import ConvCryptoTimeSeriesModel
from q_seq_model import QSequenceCryptoTimeSeriesModel
from lstm_model import LSTMCryptoTimeSeriesModel

if __name__ == "__main__":
    # Define standard variables
    datafile = "ETH-USD.csv"
    index_col =  "Date"
    y_col = 0
    writefile = "feb23.csv"
    num_train = 12
    num_test = 3
    batch_size = 3
    start_index = 100
    lookback = 8
    y_test = None
    y_preds = {}

    # Define models
    model1 = ConvCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        epochs = 4,
        lr = .005,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
        conv = [128, 64], 
        quantum=False
    )
    model2 = QSequenceCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        epochs = 4,
        lr = .005,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
    )

    # Models to try out 
    models = [model1, model2]

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
        plotfile = "{}_class_{}_ex_{}.png".format(model.type, model.y_col, i)
        model.plot(plotfile=plotfile)

    # Plot final results
    model1.plot(plotfile="Multiple_class_{}_ex_{}".format(model1.y_col, 1), y_preds=y_preds)


