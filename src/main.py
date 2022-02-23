from re import M
from conv_model import ConvCryptoTimeSeriesModel
from q_seq_model import QSequenceCryptoTimeSeriesModel

if __name__ == "__main__":
    datafile = "ETH-USD.csv"
    index_col =  "Date"
    y_col = 0
    writefile = "feb22.csv"

    if 0:
        model = ConvCryptoTimeSeriesModel(
                num_train = 800, 
                num_test = 400,
                epochs = 1000,
                lr = .005,
                batch_size = 25,
                start_index = 100,
                lookback = 7
        )
        model.read(datafile,index_col)
        model.preprocess(y_col = y_col)
        model.initialize_layers(conv = [128, 64], quantum=False)
        model.train()
        model.test()
        model.write(writefile=writefile)
        plotfile = "{}_class_{}_ex_2.png".format("quantum" if model.quantum else"classical", model.y_col)
        model.plot(plotfile=plotfile)

    model = QSequenceCryptoTimeSeriesModel(
            num_train = 800, 
            num_test = 400,
            epochs = 1000,
            lr = .005,
            batch_size = 25,
            start_index = 100,
            lookback = 14
    )

    model.read(datafile,index_col)
    model.preprocess(y_col = y_col)     
    model.initialize()
    model.train()
    model.test()
    model.write(writefile=writefile)
    plotfile = "{}_class_{}_ex_3.png".format("qcircuit", model.y_col)
    model.plot(plotfile=plotfile)