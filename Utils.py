import matplotlib.pyplot as plt

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
