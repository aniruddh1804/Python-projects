import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from Constants import *
from Utils import *

df = pd.read_csv('C:/Users/atejomurtula/Documents/MACHINE LEARNING/Loan prediction/input/train_ctrUa4K.csv')


allNumericFeatures = list(set(df.columns) - set(allCategoricalFeatures))

allNumericFeatures.remove(targetVariable)
# df['Credit_History_map'] = map_null_values(CREDIT_HISTORY,False)

def fill_null_values():
    df[GENDER].fillna('No gender', inplace=True)
    df[LOAN_AMOUNT_TERM].fillna(360.0, inplace=True)
    df[MARRIED].fillna('No', inplace=True)
    df.set_value(104, MARRIED, 'Yes')
    df[DEPENDENTS].fillna('0', inplace=True)
    df[SELF_EMPLOYED].fillna('No', inplace=True)
    df[CREDIT_HISTORY].fillna(1.0, inplace=True)
    # loan_amount = df.pivot_table(values=LOAN_AMOUNT, index=SELF_EMPLOYED, columns=[EDUCATION], aggfunc=np.median)
    # loan_amount_func = lambda x : loan_amount.loc[df[SELF_EMPLOYED], df[EDUCATION]]
    # df[LOAN_AMOUNT].fillna(df[df[LOAN_AMOUNT].isnull()].apply(loan_amount_func, axis=1), inplace=True)
    loan_amount_mean_value = df.loc[df[LOAN_AMOUNT] > 0, [LOAN_AMOUNT]].mean()[0]
    df[LOAN_AMOUNT].fillna(loan_amount_mean_value, inplace=True)


fill_null_values()

# convert categorical features to numbers (ordinal)
to_numbers_features = allCategoricalFeatures + [targetVariable]
to_numbers_features.remove(LOAN_AMOUNT_TERM)
to_numbers_features.remove(PROPERTY_AREA)
convert_data_to_numbers(to_numbers_features, df)


# transform continuous features
sc_x = StandardScaler()
df[APPLICANT_INCOME] = sc_x.fit_transform(df.ApplicantIncome.values.reshape(-1,1))
df[LOAN_AMOUNT] = sc_x.fit_transform(df.LoanAmount.values.reshape(-1,1))
df[CO_APPLICANT_INCOME] = sc_x.fit_transform(df.CoapplicantIncome.values.reshape(-1,1))

# one hot encoding non ordinal features
property_area_df = pd.get_dummies(df[PROPERTY_AREA])
df = pd.concat([df, property_area_df], axis='columns')


# machine learning
Y = df[targetVariable].to_frame()
X = df.drop([targetVariable, PROPERTY_AREA, LOAN_ID], axis=1)
model = LogisticRegression()
classification_model(model, X, Y, 0.2)

# print(X[0:4])
# print(confusion_matrix(Y_pred, Y))
# colors = np.where(df[LOAN_STATUS] == 1, 'y', 'k')
# plt.scatter(list(df[LOAN_AMOUNT]), X, c=df[LOAN_STATUS])
# plt.show()
