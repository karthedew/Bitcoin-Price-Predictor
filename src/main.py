import argparse
import os
from data_collector import DataCollector
from preprocessing import DataPreprocessing
from model import Model
from evaluate import EvaluateModel

def main():

    parser = argparse.ArgumentParser(description='Parse command-line options.')
    args   = do_options_parsing(parser=parser)

    # Setup class instances.
    dataCollector: DataCollector = DataCollector()
    if args.collect_data:
        dataCollector.collect()
        dataCollector.save()
    else:
        dataCollector.load()

    # dataCollector.check_for_na()

    # dataCollector.data_analysis()
    dataPreprocessed: DataPreprocessing = DataPreprocessing(
        data=dataCollector.df
    )
    dataPreprocessed.process()

    model = Model(
        x_data=dataPreprocessed.x_scaled_data,
        y_data=dataPreprocessed.y_scaled_data,
        x_scaler=dataPreprocessed.x_scaler,
        y_scaler=dataPreprocessed.y_scaler,
        nproc=args.nproc
    )

    if args.train:
        model.split_train_test_data()
        model.train()
        model.save()
    else:
        model.load()
        model.split_train_test_data()

    y_train_pred = model.predict(data=model.X_train)
    y_test_pred = model.predict(data=model.X_test)

    evaluateModel: EvaluateModel = EvaluateModel()

    print(model.X_train)
    print(model.y_train)

    evaluateModel.evaluate_model(y_true=model.y_test, y_pred=y_test_pred)

    train_ratio = 0.7
    train_size  = int(len(dataCollector.df) * train_ratio)
    
    train_dates = dataCollector.df.iloc[:train_size]['date']
    test_dates  = dataCollector.df.iloc[train_size:]['date']

    evaluateModel.plot_all_predictions(
        dates=dataCollector.df['date'],
        y_actual=dataCollector.df['market_price_usd'],
        train_dates=train_dates, 
        test_dates=test_dates,
        y_train_pred=y_train_pred, 
        y_test_pred=y_test_pred,
        title="Bitcoin Prediction vs. Price"
    )


def do_options_parsing(parser: argparse.ArgumentParser):
    parser.add_argument('-n', '--nproc', type=int, default=os.cpu_count()-1, help='The number of processes to use with scikit-learn (default: os.cpu_count() - 1)')
    parser.add_argument('-t', '--train', action='store_true', help='Select if you want to train the Bitcoin Predictor model.')
    parser.add_argument('-c', '--collect-data', action='store_true', help='Collect data.')
    return parser.parse_args()

if __name__ == '__main__':
    main()