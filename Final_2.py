import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm
import scipy.stats as ss
from sklearn import feature_selection as fs
import sklearn.decomposition as skde

adv_custs = pd.read_csv('AdvWorksCusts.csv')
# print(adv_custs.head())
adv_spend = pd.read_csv('AW_AveMonthSpend.csv')
# print(adv_spend.head())
adv_bike = pd.read_csv('AW_BikeBuyer.csv')
# print(adv_bike.head())
adv_custs = pd.concat([adv_custs, adv_spend['AveMonthSpend']], axis = 1)
adv_custs = pd.concat([adv_custs, adv_bike['BikeBuyer']], axis = 1)
# print(adv_custs.head())
# print(adv_custs.shape)

# removing repeated customer ID
adv_custs.drop_duplicates(subset = 'CustomerID', keep = 'first', inplace = True)
# print(adv_custs.CustomerID.unique().shape)

# print(adv_custs.dtypes)

# print(adv_custs.AveMonthSpend.min())
# print(adv_custs.AveMonthSpend.max())
# print(adv_custs.AveMonthSpend.mean())
# print(adv_custs.AveMonthSpend.median())
# print(adv_custs.AveMonthSpend.std())

# print(adv_custs.BikeBuyer.mean())

def plot_box(adv_custs, cols, col_y):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data = adv_custs)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.show()

# plot_box(adv_custs, ['Occupation'], 'YearlyIncome')

adv_custs.BirthDate = [str.replace('-', '') for str in adv_custs.BirthDate]
# adv_custs['BirthDate'] = pd.to_numeric(adv_custs['BirthDate'])

age = adv_custs['BirthDate'].str[:]
age = list(map(int, age))
# print(age)
for i in range(0, len(age), 1):
    if str(age[i])[4:] == '0101':
        age[i] = 1998 - int(str(age[i])[0:4])
    else:
        age[i] = 1998 - int(str(age[i])[0:4]) - 1

age = list(map(int, age))
adv_custs['Age'] = age
# print(adv_custs.head())
# print(adv_custs.dtypes)

def plot_scatter_shape(adv_custs, cols, shape_col = 'Gender', col_y = 'AveMonthSpend', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    unique_cats = adv_custs[shape_col].unique()
    for col in cols:
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            temp = adv_custs[adv_custs[shape_col] == cat]
            sns.regplot(col, col_y, data = temp, marker = shapes[i], label = cat, scatter_kws = {"alpha":alpha}, fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col)
        plt.xlabel(col)
        plt.ylabel(col_y)
        plt.legend()
        plt.show()

num_cols = ['Age']
# plot_scatter_shape(adv_custs, num_cols)

num_cols = ['MaritalStatus', 'NumberCarsOwned', 'Gender', 'NumberChildrenAtHome']
# plot_box(adv_custs, num_cols, 'AveMonthSpend')

def plot_box_y(adv_custs, cols, col_y):
    for col in col_y:
        sns.set_style("whitegrid")
        sns.boxplot(cols, col, data = adv_custs)
        plt.xlabel(cols)
        plt.ylabel(col)
        plt.show()

num_cols = ['YearlyIncome', 'NumberCarsOwned']
# plot_box_y(adv_custs, 'BikeBuyer', num_cols)

occupation_bike = pd.crosstab(index=adv_custs['Occupation'],columns=adv_custs['BikeBuyer'])
# print(occupation_bike)
gender_bike = pd.crosstab(index=adv_custs['Gender'],columns=adv_custs['BikeBuyer'])
# print(gender_bike)
marital_bike = pd.crosstab(index=adv_custs['MaritalStatus'],columns=adv_custs['BikeBuyer'])
# print(marital_bike)

def count_unique(adv_custs,cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(adv_custs[col].value_counts())

cols = ['StateProvinceName', 'CountryRegionName']
# count_unique(adv_custs, cols)

adv_custs.drop('BikeBuyer', axis = 1, inplace = True)

labels_train = np.array(adv_custs[['AveMonthSpend']].applymap(math.log))
adv_custs.drop('AveMonthSpend', axis = 1, inplace = True)

adv_test = pd.read_csv('AW_test.csv')
# adv_test.BirthDate = [str.replace('/', '') for str in adv_test.BirthDate]
# print(adv_test.BirthDate)

age = adv_test['BirthDate'].str[:]
# age = list(map(int, age))
# print(age)
for i in range(0, len(age), 1):
    if str(age[i])[:4] == '1/1/':
        age[i] = 1998 - int(str(age[i])[-4:])
    else:
        age[i] = 1998 - int(str(age[i])[-4:]) - 1

age = list(map(int, age))
adv_test['Age'] = age

# print(adv_test.head())
# print(adv_test.dtypes)
# print(adv_test.shape)
# print(adv_custs.shape)

frames = [adv_custs, adv_test]
adv_custs = pd.concat(frames)

adv_custs[['LogYearlyIncome']] = adv_custs[['YearlyIncome']].applymap(math.log)
# print(adv_custs.head())
# print(adv_custs.dtypes)
# print(adv_custs.shape)

Features = adv_custs['CountryRegionName']
enc = preprocessing.LabelEncoder()
enc.fit(Features)
Features = enc.transform(Features)

ohe = preprocessing.OneHotEncoder()
encoded = ohe.fit(Features.reshape(-1, 1))
Features = encoded.transform(Features.reshape(-1, 1)).toarray()

def encode_string(cat_feature):
    # encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    # apply the one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1, 1))
    return encoded.transform(enc_cat_feature.reshape(-1, 1)).toarray()

categorical_columns = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag']

for col in categorical_columns:
    temp = encode_string(adv_custs[col])
    Features = np.concatenate([Features, temp], axis = 1)

# print(Features[:10, :])
Features = np.concatenate([Features, np.array(adv_custs[['Age', 'LogYearlyIncome', 'NumberCarsOwned',
                                                         'NumberChildrenAtHome', 'TotalChildren']])], axis = 1)

Features_train = Features[:16404]
Features_final = Features[-500:]
# print(Features_test.shape)
# print(Features_train.shape)
# print(labels_train.shape)

nr.seed(9988)
indx = range(Features_train.shape[0])
indx = ms.train_test_split(indx, test_size = 1000)
x_train = Features_train[indx[0],:]
y_train = np.ravel(labels_train[indx[0]])
x_test = Features_train[indx[1],:]
y_test = np.ravel(labels_train[indx[1]])

scaler = preprocessing.StandardScaler().fit(x_train[:, -5:])
x_train[:, -5:] = scaler.transform(x_train[:, -5:])
x_test[:, -5:] = scaler.transform(x_test[:, -5:])

lin_mod = linear_model.LinearRegression(fit_intercept = False)
lin_mod.fit(x_train, y_train)

def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(sklm.mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(sklm.median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))

def resid_plot(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1, 1), y_score.reshape(-1, 1))
    ## now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

def hist_resids(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1, 1), y_score.reshape(-1, 1))
    ## now make the residual plots
    sns.distplot(resids)
    plt.title('Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('count')
    plt.show()

def resid_qq(y_test, y_score):
    ## first compute vector of residuals.
    resids = np.subtract(y_test, y_score)
    ## now make the residual plots
    ss.probplot(resids.flatten(), plot=plt)
    plt.title('Residuals vs. predicted values')
    plt.xlabel('Predicted values')
    plt.ylabel('Residual')
    plt.show()

y_score = lin_mod.predict(x_test)
print_metrics(y_test, y_score)
# hist_resids(y_test, y_score)
# resid_qq(y_test, y_score)
# resid_plot(y_test, y_score)

# l2 regularization
def plot_regularization(l, train_RMSE, test_RMSE, coefs, min_idx, title):
    plt.plot(l, test_RMSE, color='red', label='Test RMSE')
    plt.plot(l, train_RMSE, label='Train RMSE')
    plt.axvline(min_idx, color='black', linestyle='--')
    plt.legend()
    plt.xlabel('Regularization parameter')
    plt.ylabel('Root Mean Square Error')
    plt.title(title)
    plt.show()

    plt.plot(l, coefs)
    plt.axvline(min_idx, color='black', linestyle='--')
    plt.title('Model coefficient values \n vs. regularizaton parameter')
    plt.xlabel('Regularization parameter')
    plt.ylabel('Model coefficient value')
    plt.show()

def test_regularization_l2(x_train, y_train, x_test, y_test, l2):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l2:
        lin_mod = linear_model.Ridge(alpha=reg)
        lin_mod.fit(x_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(x_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(x_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l2 = l2[min_idx]
    min_RMSE = test_RMSE[min_idx]

    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l2, train_RMSE, test_RMSE, coefs, min_l2, title)
    return min_l2, min_RMSE

l2 = [x for x in range(1, 101)]
out_l2 = test_regularization_l2(x_train, y_train, x_test, y_test, l2)
# print(out_l2)

lin_mod_l2 = linear_model.Ridge(alpha = out_l2[0])
lin_mod_l2.fit(x_train, y_train)
y_score_l2 = lin_mod_l2.predict(x_test)
print_metrics(y_test, y_score_l2)
hist_resids(y_test, y_score_l2)
resid_qq(y_test, y_score_l2)
resid_plot(y_test, y_score_l2)
# only slightly better, not much improvement

# l1 regularization
def test_regularization_l1(x_train, y_train, x_test, y_test, l1):
    train_RMSE = []
    test_RMSE = []
    coefs = []
    for reg in l1:
        lin_mod = linear_model.Lasso(alpha=reg)
        lin_mod.fit(x_train, y_train)
        coefs.append(lin_mod.coef_)
        y_score_train = lin_mod.predict(x_train)
        train_RMSE.append(sklm.mean_squared_error(y_train, y_score_train))
        y_score = lin_mod.predict(x_test)
        test_RMSE.append(sklm.mean_squared_error(y_test, y_score))
    min_idx = np.argmin(test_RMSE)
    min_l1 = l1[min_idx]
    min_RMSE = test_RMSE[min_idx]

    title = 'Train and test root mean square error \n vs. regularization parameter'
    plot_regularization(l1, train_RMSE, test_RMSE, coefs, min_l1, title)
    return min_l1, min_RMSE

l1 = [x / 5000 for x in range(1, 100)]
out_l1 = test_regularization_l1(x_train, y_train, x_test, y_test, l1)
# print(out_l1)

lin_mod_l1 = linear_model.Lasso(alpha = out_l1[0])
lin_mod_l1.fit(x_train, y_train)
y_score_l1 = lin_mod_l1.predict(x_test)
print_metrics(y_test, y_score_l1)
hist_resids(y_test, y_score_l1)
resid_qq(y_test, y_score_l1)
resid_plot(y_test, y_score_l1)
# slightly worse performance than l2 regularization

scaler = preprocessing.StandardScaler().fit(Features_train[:, -5:])
Features_final[:, -5:] = scaler.transform(Features_final[:, -5:])

y_score_l2 = lin_mod_l2.predict(Features_final)
scores = np.exp(y_score_l2)

adv_test['AveMonthSpend'] = scores

# print(adv_test.head())
adv_test.to_csv('ADV_AveMonthSpend_Predictions.csv', index = False, header = True)




