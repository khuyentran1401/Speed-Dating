import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import joblib

from pipeline import pipeline
import hydra
from hydra import utils
import os



@hydra.main(config_path='experiments/preprocessing.yaml')
def run_training(config):
    """Train the model."""

    #print(os.getcwd())
    current_path = utils.get_original_cwd() + "/"
    # read training data
    data = pd.read_csv(current_path + config.dataset.data,  encoding=config.dataset.encoding)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.target.target, axis=1),
        data[config.target.target],
        test_size=0.1,
        random_state=0)  # we are setting the seed here
    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)

    # transform the target
    #y_train = np.log(y_train)
    #y_test = np.log(y_test)

    match_pipe = pipeline(config)
    match_pipe.fit(X_train, y_train)
    joblib.dump(match_pipe, utils.to_absolute_path(config.pipeline.pipeline01))


if __name__ == '__main__':
    run_training()