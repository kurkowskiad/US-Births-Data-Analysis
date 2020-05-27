import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition, preprocessing

def train_logistic_regression(solver, max_iter, multi_class,
                              x_train, y_train,
                              x_test, y_test):
    """Uczenie klasyfikatora regresji logistycznej,
    porównanie otrzymanej dokładności na zbiorze treningowym i testowym"""

    model = LogisticRegression(solver=solver,
                               max_iter=max_iter,
                               multi_class=multi_class)
    model.fit(x_train, y_train)
    lr_train_accuracy = model.score(x_train, y_train)
    lr_test_accuracy = model.score(x_test, y_test)
    print('Dokladnosc dla zbioru treningowego regresji logistycznej: {0}.'.format(lr_train_accuracy))
    print('Dokladnosc dla zbioru testowego regresji logistycznej: {0}'.format(lr_test_accuracy))
    return model


def train_random_forest(n_estimators, max_leaf_nodes, n_jobs,
                        x_train, y_train,
                        x_test, y_test):
    """Uczenie klasyfikatora lasu losowego,
    porównanie otrzymanej dokładności na zbiorze treningowym i testowym"""

    forest = RandomForestClassifier(n_estimators=n_estimators,
                                    max_leaf_nodes=max_leaf_nodes,
                                    n_jobs=n_jobs).fit(x_train,
                                                       y_train)
    forest_train_accuracy = forest.score(x_train, y_train)
    forest_test_accuracy = forest.score(x_test, y_test)
    print('Dokladnosc dla zbioru treningowego lasu losowego: {0}.'.format(forest_train_accuracy))
    print('Dokladnosc dla zbioru testowego lasu losowego: {0}.'.format(forest_test_accuracy))
    return forest


cols_to_keep = ['BFACIL', 'BMI', 'CIG_0', 'DBWT',
                'DOB_MM', 'DOB_TT', 'DOB_WK', 'DWgt_R',
                'FAGECOMB', 'MAGER', 'M_Ht_In', 'DMAR',
                'NO_RISKS', 'PREVIS', 'PWgt_R', 'SEX', 'WTGAIN']
dtypes = \
    {'BFACIL': np.uint8, 'BMI': np.float32, 'CIG_0': np.uint8, 'DBWT': np.uint16,
     'DMAR': np.uint8, 'DOB_MM': np.uint8, 'DOB_TT': np.uint16, 'DWgt_R': np.uint16,
     'DOB_WK': np.uint8, 'FAGECOMB': np.uint8, 'MAGER': np.uint8, 'M_Ht_In': np.uint8,
     'NO_RISKS': np.uint8, 'PREVIS': np.uint8, 'PWgt_R': np.uint16, 'SEX': np.uint8,
     'WTGAIN': np.uint8}
pandas_data = pandas.read_csv("US_births(2018).csv", delimiter=',', usecols=cols_to_keep, dtype=None)
pandas_data.loc[pandas_data['SEX'] == 'M', 'SEX'] = 0
pandas_data.loc[pandas_data['SEX'] == 'F', 'SEX'] = 1
pandas_data = pandas_data.replace(r'^\s*$', 0, regex=True)
pandas_data = pandas_data.astype(dtypes)
x1 = pandas_data.iloc[:, 0:15]
x2 = pandas_data.iloc[:, 16]
x = pandas.concat((x1, x2), axis=1)
y = pandas_data.iloc[:, 15]
# Podzielenie zbioru danych na zbiór treningowy i testowy 60/40.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.3)
pca = decomposition.PCA(0.95)
principalComponents = pca.fit_transform(x)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# Uczenie klasyfikatora regresji logistycznej
lr_model = train_logistic_regression(solver='lbfgs', max_iter=5000, multi_class='multinomial',
                                     x_train=x_train, y_train=y_train,
                                    x_test=x_test, y_test=y_test)
#forest = train_random_forest(n_estimators=30, max_leaf_nodes=10, n_jobs=-1,
#                             x_train=x_train, y_train=y_train,
#                             x_test=x_test, y_test=y_test)

# MemoryError: Unable to allocate 87.9 GiB for an array with shape (2280920, 5170) and data type int64
