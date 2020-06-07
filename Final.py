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

adv_custs.drop('AveMonthSpend', axis = 1, inplace = True)

labels_train = np.array(adv_custs['BikeBuyer'])
adv_custs.drop('BikeBuyer', axis = 1, inplace = True)


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

# print(adv_custs.head())
# print(adv_custs.dtypes)
# print(adv_custs.shape)



adv_custs[['LogYearlyIncome']] = adv_custs[['YearlyIncome']].applymap(math.log)
# print(adv_custs.head())

Features = adv_custs['StateProvinceName']
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

categorical_columns = ['Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag',
                       'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren']

for col in categorical_columns:
    temp = encode_string(adv_custs[col])
    Features = np.concatenate([Features, temp], axis = 1)

# print(Features[:10, :])
Features = np.concatenate([Features, np.array(adv_custs[['Age', 'LogYearlyIncome']])], axis = 1)

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

scaler = preprocessing.StandardScaler().fit(x_train[:, -2:])
x_train[:, -2:] = scaler.transform(x_train[:, -2:])
x_test[:, -2:] = scaler.transform(x_test[:, -2:])

print(x_train)
print(y_train)
print(x_test)
print(y_test)

logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(x_train, y_train)

probabilities = logistic_mod.predict_proba(x_test)
# print(probabilities)

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

# print_metrics(y_test, scores)
# class imbalance in the data is confirmed

def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:, 1])
    auc = sklm.auc(fpr, tpr)

    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# plot_auc(y_test, probabilities)

pca_mod = skde.PCA()
pca_comps = pca_mod.fit(x_train)

def plot_explained(mod):
    comps = mod.explained_variance_ratio_
    x = range(len(comps))
    x = [y + 1 for y in x]
    plt.plot(x, comps)
    plt.show()

# plot_explained(pca_comps)

# create a subset of the 30 best features
pca_mod_40 = skde.PCA(n_components = 40)
pca_mod_40.fit(x_train)
Comps = pca_mod_40.transform(x_train)

# logistic regression model with the pca
log_mod_40 = linear_model.LogisticRegression(C = 10.0, class_weight = {0:0.1, 0:0.9})
log_mod_40.fit(Comps, y_train)
# print(log_mod_40.intercept_)
# print(log_mod_40.coef_)


def print_metrics_2(labels, probs, threshold):
    scores = score_model(probs, threshold)
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))
    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:, 1]))
    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1])) / 2.0))
    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1])) / 2.0))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


probabilities = log_mod_40.predict_proba(pca_mod_40.transform(x_test))
# print_metrics_2(y_test, probabilities, 0.3)
# plot_auc(y_test, probabilities)

# create a subset of the 20 best features
pca_mod_20 = skde.PCA(n_components = 20)
pca_mod_20.fit(x_train)
Comps = pca_mod_20.transform(x_train)

# logistic regression model with the pca
log_mod_20 = linear_model.LogisticRegression(C = 10.0, class_weight = {0:0.1, 0:0.9})
log_mod_20.fit(Comps, y_train)
# print(log_mod_20.intercept_)
# print(log_mod_20.coef_)

probabilities = log_mod_20.predict_proba(pca_mod_20.transform(x_test))
# print_metrics_2(y_test, probabilities, 0.3)
# plot_auc(y_test, probabilities)

def print_format(f,x,y,z):
    print('Fold %2d    %4.3f        %4.3f      %4.3f' % (f, x, y, z))

def print_cv(scores):
    fold = [x + 1 for x in range(len(scores['test_precision_macro']))]
    print('         Precision     Recall       AUC')
    [print_format(f,x,y,z) for f,x,y,z in zip(fold, scores['test_precision_macro'],
                                          scores['test_recall_macro'],
                                          scores['test_roc_auc'])]
    print('-' * 40)
    print('Mean       %4.3f        %4.3f      %4.3f' %
          (np.mean(scores['test_precision_macro']), np.mean(scores['test_recall_macro']), np.mean(scores['test_roc_auc'])))
    print('Std        %4.3f        %4.3f      %4.3f' %
          (np.std(scores['test_precision_macro']), np.std(scores['test_recall_macro']), np.std(scores['test_roc_auc'])))

labels_train = labels_train.reshape(labels_train.shape[0], )
scoring = ['precision_macro', 'recall_macro', 'roc_auc']

pca_mod = skde.PCA(n_components = 20)
pca_mod.fit(Features_train)
Comps = pca_mod.transform(Features_train)

scores = ms.cross_validate(log_mod_20, Comps, labels_train, scoring=scoring,
                        cv=10, return_train_score=False)
# print_cv(scores)

pca_mod = skde.PCA(n_components = 40)
pca_mod.fit(Features_train)
Comps = pca_mod.transform(Features_train)

scores = ms.cross_validate(log_mod_40, Comps, labels_train, scoring=scoring,
                        cv=10, return_train_score=False)
# print_cv(scores)

final_prediction = log_mod_40.predict_proba(pca_mod_40.transform(Features_final))
# print(final_prediction)

scaler = preprocessing.StandardScaler().fit(Features_train[:, -2:])
Features_final[:, -2:] = scaler.transform(Features_final[:, -2:])
predictions = log_mod_40.predict_proba(pca_mod_40.transform(Features_final))
scores = score_model(predictions, 0.5)
# print(scores)

# print(adv_test.shape)
# print(len(scores))
adv_test['BikeBuyer'] = scores

# print(adv_test.head())
adv_test.to_csv('ADV_Bike_Predictions.csv', index = False, header = True)