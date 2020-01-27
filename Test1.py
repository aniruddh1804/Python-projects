import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np

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



# _______________DATA UNDERSTANDING______________________
# print(df.pivot_table(values=targetVariable, columns=['applicant_income_median'], aggfunc=np.sum))
# df[GENDER] = df[GENDER].map({'Male': 2, 'Female': 0, 'No gender': 1})
# df[MARRIED] = df[MARRIED].map({'No': 1, 'Yes': 0})
# df[EDUCATION] = df[EDUCATION].map({'Graduate': 1, 'Not Graduate': 0})
# df[SELF_EMPLOYED] = df[SELF_EMPLOYED].map({'Yes' : 0, 'No' : 1})
df[targetVariable] = df[targetVariable].map({'Y': 1, 'N': 0})


# print(df['applicant_income_median'].head())

# df[LOAN_AMOUNT+LOG] = np.log(df[LOAN_AMOUNT])
# histogram(df, LOAN_AMOUNT, 20, np.nan)

# le = LabelEncoder()
#
# for col in allCategoricalFeatures:
#     df[col] = le.fit_transform(df[col])

Y = df[targetVariable].to_frame()

sc_x = StandardScaler()
df[APPLICANT_INCOME] = pd.DataFrame(np.log(df[APPLICANT_INCOME]))
X = sc_x.fit_transform(df.ApplicantIncome.values.reshape(-1,1))
model = LogisticRegression()
model.fit(X, Y)
Y_pred = model.predict(X)
print(X[0:4])
print(confusion_matrix(Y_pred, Y))
colors = np.where(df[LOAN_STATUS] == 1, 'y', 'k')
plt.scatter(list(df[LOAN_AMOUNT]), list(X), c=df[LOAN_STATUS])
plt.show()
