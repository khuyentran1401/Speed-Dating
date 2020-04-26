import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna('Missing')

        return X


# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        
        self.variables = []
            

    def fit(self, X, y=None):
        
        self.variables = [var for var in X.columns if X[var].dtype != 'O']
        
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self, X):

        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X


# Temporal variable calculator
'''
class TemporalVariableEstimator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, reference_variable=None):
        
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

        self.reference_variables = reference_variable

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variables] - X[feature]

        return X
'''

# transform categorical to numerical
class CategoricalToNumerical(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):

        self.variables = variables

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        def object_to_num(x):

            if x is not np.nan:
                x = ''.join(x.split(','))

            return x
        
        for feature in self.variables:

            X[feature] = X[feature].apply(object_to_num)
            X[feature] = X[feature].astype(np.float)
        return X


# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        
        self.tol = tol
        
        self.variables = variables

    def fit(self, X, y=None):

        # persist frequent labels in dictionary
        self.encoder_dict_ = {}

        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)

        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[
                    feature]), X[feature], 'Rare')

        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None, target=None):
        
        self.variables = variables

    def fit(self, X, y=None):
         
        self.enc = OneHotEncoder(handle_unknown='ignore')
        # persist transforming dictionary
        

        if len(self.variables) == 1:
            self.enc.fit(np.array(X[self.variables[0]]).reshape(-1,1))

        else:
            self.enc.fit(np.array(X[self.variables]))

        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        
        if len(self.variables) == 1:
            X_enc = self.enc.transform(np.array(X[self.variables[0]]).reshape(-1,1)).toarray()


        else:
            X_enc = self.enc.transform(np.array(X[self.variables])).toarray()


        X_enc = pd.DataFrame(X_enc, columns = [self.variables[j] + '_' + str(i) for j in range(len(self.variables)) for i in range(len(self.enc.categories_[j]))])
        
        X = pd.concat([X, X_enc], axis=1).drop(self.variables, axis=1)
            
            
        
        return X


# logarithm transformer
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = []

    def fit(self, X, y=None):
        # to accomodate the pipeline
        self.variables = [var for var in X.columns if X[var].dtype != 'O']

        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            X[feature] = np.log(X[feature])

        return X


class DropUnecessaryFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, variables_to_drop=None):

        self.variables = variables_to_drop

    def fit(self, X, y=None):
        

        return self


    def transform(self, X):
        # encode labels
        X = X.copy()
        X = X.drop(self.variables, axis=1)

        return X

class LabelExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        for feature in self.variables:
            for i in range(X.shape[0]):
                try:
                    X[feature][i] = X[feature][i].split(',')[0]
                except:
                    pass

        return X






