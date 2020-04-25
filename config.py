# data
TRAINING_DATA_FILE = "data/raw.csv"
PIPELINE_NAME = 'classification'

TARGET = 'match'

# this variable is to calculate the temporal variable,
# must be dropped afterwards
DROP_FEATURES = ['iid','id','idg','wave','position','positin1', 'pid',  'field', 'from', 'career']

# categorical variables to transform to numerical variables
NUMERICAL_VARS_FROM_CATEGORICAL = ['income','mn_sat', 'tuition']

# categorical variables to encode
CATEGORICAL_VARS = ['undergra', 'zipcode']
CATEGORICAL_LABEL_EXTRACTION = 'zipcode'
CATEGORICAL_ONEHOT = 'undergra'
