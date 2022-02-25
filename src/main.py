"""
===================================================
MAIN.PY
===================================================
"""
# Imports
from conv_model import ConvCryptoTimeSeriesModel
from q_seq_model import QSequenceCryptoTimeSeriesModel

if __name__ == "__main__":
    # Define standard variables
    datafile = "ETH-USD.csv"
    index_col =  "Date"
    y_col = 0
    writefile = "feb25.csv"
    num_train = 100
    num_test = 50
    batch_size = 5
    start_index = 200
    lookback = 4
    lr = .005
    y_test = None
    y_preds = {}

    # Define models
    model1 = ConvCryptoTimeSeriesModel(
        num_train = num_train, 
        num_test = num_test,
        iterations = 1000,
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
        iterations = 1000,
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
        iterations = 100,
        lr = lr,
        batch_size = batch_size,
        start_index = start_index,
        lookback = lookback,
        num_layers = 2,
    )
    model4 = QSequenceCryptoTimeSeriesModel(
        num_train = 20, 
        num_test = 10,
        iterations = 1,
        lr = lr,
        batch_size = 5,
        start_index = 0,
        lookback = 4,
        num_layers = 2,
    )
    model5 = ConvCryptoTimeSeriesModel(
        num_train = 20, 
        num_test = 10,
        iterations = 1,
        lr = lr,
        batch_size = 5,
        start_index = 0,
        lookback = 4,
        conv = [64, 32], 
        quantum=True
    )
    # Models to try out 
    # models = [model1, model2]
    # models = [model3]
    models = [model1, model2]

    # Run each model
    for i in range(len(models)):
        model = models[i]
        model.read(datafile,index_col)
        model.preprocess(y_col = y_col)
        if type(model) == ConvCryptoTimeSeriesModel: model.initialize_layers()
        model.train()
        model.test()
        y_test, y_pred_i = model.invtransform_y()
        model.write(writefile=writefile)
        if i > 0: y_preds[model.type] = y_pred_i
        plotfile = "{}_class_{}_start_{}.png".format(model.type, model.y_col, start_index)
        model.plot(plotfile=plotfile)

    # Plot final results
    if len(models) > 1:
        model1.plot(plotfile="Multiple_class_{}_index_{}".format(model1.y_col, start_index), y_preds=y_preds)


