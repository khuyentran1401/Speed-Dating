# data
TRAINING_DATA_FILE = "data/raw.csv"
PIPELINE_NAME = 'classification'

TARGET = 'match'

# input variables 
'''
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage',
            # this one is only to calculate temporal variable:
            'YrSold']
'''
# this variable is to calculate the temporal variable,
# must be dropped afterwards
DROP_FEATURES = ['iid','id','idg','wave','position','positin1', 'pid',  'field', 'from', 'career']

# categorical variables to transform to numerical variables
NUMERICAL_VARS_FROM_CATEGORICAL = ['income','mn_sat', 'tuition']

# numerical variables with NA in train set
#NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables with NA in train set
# CATEGORICAL_VARS_WITH_NA 

#T EMPORAL_VARS = 

# variables to log transform
# NUMERICALS_LOG_VARS = ['Interest correlation','age of partner', 'income', 'mn_sat','tuition']

# categorical variables to encode
CATEGORICAL_VARS = ['undergra', 'zipcode']
CATEGORICAL_LABEL_EXTRACTION = 'zipcode'
CATEGORICAL_ONEHOT = 'undergra'