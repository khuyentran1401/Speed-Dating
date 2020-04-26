import pandas as pd

import joblib

from sklearn.metrics import confusion_matrix
import graphviz
from sklearn import tree
import hydra

# test pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from hydra import utils


def make_prediction(input_data, config):
    
    _pipe_match = joblib.load(filename=utils.to_absolute_path(config.pipeline.pipeline01))
    
    results = _pipe_match.predict(input_data)

    return results, _pipe_match
   

@hydra.main(config_path='experiments/preprocessing.yaml')
def training(config):   

    current_path = utils.get_original_cwd() + "/"

    data = pd.read_csv(current_path + config.dataset.data, encoding=config.dataset.encoding)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.target.target, axis=1),
        data[config.target.target],
        test_size=0.1,
        random_state=0)

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    
    pred, pipe = make_prediction(X_test, config)
    
    # determine mse and rmse
    print('test mse: {}'.format(int(
        mean_squared_error(y_test, np.exp(pred)))))
    print('test rmse: {}'.format(int(
        np.sqrt(mean_squared_error(y_test, np.exp(pred))))))
    print('test r2: {}'.format(
        r2_score(y_test, np.exp(pred))))
    print('confusion matrix: {}'.format(
        confusion_matrix(y_test, pred)))

    dot_data = tree.export_graphviz(pipe['decision tree classifier'], out_file=None)
    graph =graphviz.Source(dot_data)
    graph.render('tree')

    print(X_train.columns[84])
    print(X_train.columns[15])


if __name__ == '__main__':
    training()