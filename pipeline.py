from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.tree import DecisionTreeClassifier

import preprocessors as pp
import config


match_pipe = Pipeline(

    [
        ('drop_fatures',
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES)),
        
        ('categorical_to_numerical',
            pp.CategoricalToNumerical(variables=config.NUMERICAL_VARS_FROM_CATEGORICAL)),

        ('numerical_imputer',
            pp.NumericalImputer()),

        ('categorical_imputer',
           pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
         
        #('temporal_variable',
        #    pp.TemporalVariableEstimator(
        #       variables=config.TEMPORAL_VARS,
        #        reference_variable=config.DROP_FEATURES)),
        ('label extraction',
            pp.LabelExtraction(variables=config.CATEGORICAL_LABEL_EXTRACTION)), 
        
        ('rare label encoder',
            pp.RareLabelCategoricalEncoder(
                tol=0.01,
                variables=config.CATEGORICAL_VARS)),
         
        ('categorical_encoder',
           pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
         
        #('feature hashing', 
         #   FeatureHasher(n_features=10, input_type='string')),

        #('log_transformer',
        #    pp.LogTransformer()),        
         
        ('scaler', MinMaxScaler()),

        ('decision tree classifier', DecisionTreeClassifier())
    ]
)
