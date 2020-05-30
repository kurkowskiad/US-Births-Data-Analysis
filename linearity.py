import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import seaborn
from sklearn.model_selection import train_test_split

cols_to_keep = ['ATTEND', 'BFACIL', 'BMI', 'CIG_0', 'DBWT',
                'DLMP_MM', 'DLMP_YY', 'DMAR', 'DOB_MM', 'DOB_TT',
                'DOB_WK', 'DWgt_R', 'FAGECOMB', 'FRACE6', 'MAGER',
                'MRAVE6', 'M_Ht_In', 'NO_INFEC', 'NO_RISKS', 'PRIORDEAD',
                'PRIORLIVE', 'PRECARE', 'PREVIS', 'PWgt_R', 'RDMETH_REC',
                'RF_CESARN', 'SEX', 'WTGAIN']

dtypes = \
    {'ATTEND': np.uint8, 'BFACIL': np.uint8, 'BMI': np.float32, 'CIG_0': np.uint8, 'DBWT': np.uint16,
    'DLMP_MM': np.uint8, 'DLMP_YY': np.uint16, 'DMAR': np.uint8, 'DOB_MM': np.uint8, 'DOB_TT': np.uint16,
     'DOB_WK': np.uint8, 'DWgt_R': np.uint16, 'FAGECOMB': np.uint8, 'FRACE6': np.uint8, 'MAGER': np.uint8,
     'MRAVE6': np.uint8, 'M_Ht_In': np.uint8, 'NO_INFEC': np.uint8, 'NO_RISKS': np.uint8, 'PRIORDEAD': np.uint8,
     'PRIORLIVE': np.uint8, 'PRECARE': np.uint8, 'PREVIS': np.uint8, 'PWgt_R': np.uint16, 'RDMETH_REC': np.uint8,
     'RF_CESARN': np.uint8, 'SEX': np.uint8, 'WTGAIN': np.uint8}

df = pandas.read_csv("US_births(2018).csv", delimiter=',', usecols=cols_to_keep, dtype=None)
df.loc[df['SEX'] == 'M', 'SEX'] = 0
df.loc[df['SEX'] == 'F', 'SEX'] = 1

df = df.replace(r'^\s*$', 99, regex=True)
df = df.astype(dtypes)

for column in range(0, len(df.columns)):
    df = df[df[df.columns[column]] != 99]
for column in [0, 1, 7, 13, 15, 17, 18, 19, 21, 22, 24, 25, 26]:
    df = df[df[df.columns[column]] != 9]
for column in [4, 6]:
    df = df[df[df.columns[column]] != 9999]


def plot(x_label, y_label, sample, alpha):
    smp = df.sample(sample)
    x = smp[x_label]
    y = smp[y_label]
    slope, intercept, r_value, p_value, sterr = stats.linregress(x, y)
    print(r_value ** 2)
    plt.scatter(x, y, color='red', s=4, alpha=alpha)
    pf = np.polyfit(x, y, 1)
    plt.plot(x, np.polyval(pf, x))
    plt.grid(True)
    plt.show()
# plt.title('Relation between mother BMI and child weight', fontsize=14)
# plt.xlabel('BMI', fontsize=14)
# plt.ylabel('Child weight', fontsize=14)

# sample = df.sample(5000, random_state=10)
# plt.scatter(sample['BMI'], sample['DBWT'], color='red', s=5)
# plt.grid(True)
# plt.show()

# bins = np.arange(2016,2020) + 0.5
# # df.DLMP_YY.plot(kind='hist', color='blue', edgecolor='black',alpha=.7, rwidth=.6, bins=bins)
# # plt.title('Płeć noworodka', size=20)
# # plt.xlabel('Płeć', size=18)
# # plt.ylabel('Ilość wystąpień', size=18)
# # plt.xlim((2016, 2020))
# # plt.xticks(bins - 0.5)
# # plt.show()

def label_race (row):
    months = 0
    if row['DLMP_YY'] == 2017:
        months = row['DOB_MM'] + 12
    if row['DLMP_YY'] == 2018:
        months = row['DOB_MM']
    else:
        return 99
    return months - row['DLMP_MM']


df['GST_TIME'] = df.apply(lambda row: label_race(row), axis=1)
df = df.drop(df[df['GST_TIME'] == 99].index)

# bins = np.arange(0,16) + 0.5
# df.GST_TIME.plot(kind='hist', color='blue', edgecolor='black',alpha=.7, rwidth=.6, bins=bins)
# plt.title('Płeć noworodka', size=20)
# plt.xlabel('Płeć', size=18)
# plt.ylabel('Ilość wystąpień', size=18)
# plt.xlim((0, 16))
# plt.xticks(bins - 0.5)
# plt.show()

#plot("GST_TIME", "DBWT", 100000, .05)
#smp = df.loc[df['GST_TIME'] <= 9]
pandas.set_option("display.max_columns", None)
print(df.loc[(df['GST_TIME'] == 1) & (df['DBWT'] > 200)])

x = df['GST_TIME']
y = df['DBWT']
slope, intercept, r_value, p_value, sterr = stats.linregress(x, y)
print(r_value ** 2)
splot = seaborn.stripplot(x=x, y=y, size=3, alpha=.1)
splot.set(ylim=(0, None))
#plt.scatter(x, y, color='red', s=4, alpha=.05)
pf = np.polyfit(x, y, 1)
plt.plot(x, np.polyval(pf, x))
plt.grid(True)
plt.show()
# sample = df.sample(500000)
# plot('BMI', 'DBWT', 500000, .01)
# print(df['BMI'].ndim)
# b = df['BMI'].values.reshape(-1,1)
# df['BMI'] = pandas.DataFrame(b)
#x = np.reshape(df['BMI'].values, (-1, 1))
#y = np.reshape(df['DBWT'].values, (-1,1))
# sample = df.sample(10000)
# x = sample['DWgt_R']
# y = sample['DBWT']
# slope, intercept, r_value, p_value, sterr = stats.linregress(x, y)
# print(r_value**2)
# plt.scatter(x, y, color='red', s=4, alpha=.05)
# plt.grid(True)
# plt.show()
# correlation_matrix = df.corr()
# seaborn.heatmap(correlation_matrix, annot=True)
# plt.show()

# smpl = df.sample(1000)
# smpl.plot(kind='scatter', x='BMI', y='M_Ht_In', c='DBWT', cmap=plt.get_cmap('jet'), colorbar=True)
# plt.xlabel = 'BMI'
# plt.ylabel = "M_Ht_In"
# plt.show()

def train_linear_regression(x_train, y_train,
                              x_test, y_test):
    """Uczenie klasyfikatora regresji logistycznej,
    porównanie otrzymanej dokładności na zbiorze treningowym i testowym"""

    model = LinearRegression()
    model.fit(x_train, y_train)
    lr_train_accuracy = model.score(x_train, y_train)
    lr_test_accuracy = model.score(x_test, y_test)
    print('Dokladnosc dla zbioru treningowego regresji logistycznej: {0}.'.format(lr_train_accuracy))
    print('Dokladnosc dla zbioru testowego regresji logistycznej: {0}'.format(lr_test_accuracy))
    return model


x = df["GST_TIME"]
y = df['DBWT']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
