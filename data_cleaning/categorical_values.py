# function gets a list of categorical variables
from sklearn.preprocessing import OrdinalEncoder


def getCategorialVar(X_train):
    s = (X_train.dtypes == 'object')  # 'object' shows that there is text
    return list(s[s].index)


# function that drops all columns with categorical variables

def dropCategoricalData(X_train, X_valid):
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])
    return drop_X_train, drop_X_valid


# function that uses sklearns OrdinalEncoder to get ordinal encodings.
# We loop over the cat. variables and apply the ordinal encoder separately to each column
# from sklearn.preprocessing import

def encodeOrdinal(X_train_copy, X_valid_copy):
    # Apply ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    catVar = getCategorialVar(X_train_copy)
    X_train_copy[catVar] = ordinal_encoder.fit_transform(X_train_copy[catVar])
    X_valid_copy[catVar] = ordinal_encoder.transform(X_valid_copy[catVar])
    return X_train_copy, X_valid_copy

