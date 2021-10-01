

# function that deletes all columns containing a NULL value

def deleteColumns(X_train, X_valid):
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
    return X_train.drop(cols_with_missing, axis=1), X_valid.drop(cols_with_missing, axis=1)
