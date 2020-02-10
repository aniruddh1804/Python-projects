import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

def print_all_categorical_feature_values(features, dataframe):
    for feature in features:
        print(dataframe[feature].value_counts())


def print_missing_values(dataframe):
    print(dataframe.apply(lambda x: sum(x.isnull()),axis=0))
    print("\nTotal values : ", len(dataframe))


def histogram(dataframe, column, bins, filter_column):
    if is_nan(filter_column):
        dataframe.hist(column=column, bins=bins)
    else:
        dataframe.hist(column=column, by=filter_column, bins=bins)
    plt.show()


def pie_chart(dataframe, column):
    dataframe[column].fillna('NAN', inplace=True)
    plt.pie(dataframe[column].value_counts(),autopct='%1.1f%%', shadow=False, startangle=0, labels=dataframe[column].unique())
    plt.show()


def map_null_values(column, is_integer, dataframe):
    if is_integer:
        return list(map(lambda val: map_nan_value_to_integer(val), dataframe[column]))
    else:
        return list(map(lambda val: map_nan_value_to_string(val), dataframe[column]))


def is_nan(val):
    return val != val


def map_nan_value_to_string(val):
    if is_nan(val):
        return "Nan string"
    else:
        return val


def map_nan_value_to_integer(val):
    if is_nan(val):
        return 0.0
    else:
        return val


def return_list_of_null_value_row_numbers(column, dataframe):
    return list(dataframe[dataframe[column].isnull()].index)


def prob_by_bayes_thm(column, value, dataframe, target_variable, target_variable_value):
    len_A = len(dataframe.loc[dataframe[target_variable] == target_variable_value])
    p_A = len_A / 614
    p_B = len(dataframe.loc[dataframe[column] == value]) / 614
    p_B_A = len(dataframe.loc[(dataframe[column] == value) & (dataframe[target_variable] == target_variable_value)]) / len_A
    return (p_A * p_B_A) / p_B


def convert_data_to_numbers(features, dataframe):
    le = LabelEncoder()
    for col in features:
        dataframe[col] = le.fit_transform(dataframe[col])


def classification_model(model, X, Y, test_size):
    model.fit(X, Y)
    Y_pred = model.predict(X)
    print("Training data accuracy")
    print(metrics.accuracy_score(Y_pred, Y)*100)


    print("Cross validation")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    print("Shape of training data")
    print(X_train.shape, Y_train.shape)

    print("Shape of test data")
    print(X_test.shape, Y_test.shape)

    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(metrics.accuracy_score(Y_pred, Y_test)*100)
