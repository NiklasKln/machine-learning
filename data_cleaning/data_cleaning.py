import pandas as pd
from sklearn.impute import SimpleImputer


# function to get names of columns with missing values

def getMissingColumns(X_train):
    return [col for col in X_train.columns if X_train[col].isnull().any()]


# function that deletes all COLUMNS containing a NULL value

def deleteColumns(X_train, X_valid):
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
    return X_train.drop(cols_with_missing, axis=1), X_valid.drop(cols_with_missing, axis=1)


# usage of SimpleImputer to impute missing values and add a new column that shows the
# location of the imputed entries for each column.
# from sklearn.impute import Simple Imputer

def impute(X_train, X_valid):
    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    return imputed_X_train, imputed_X_valid



