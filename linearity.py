import pandas
import numpy as np
import matplotlib.pyplot as plt

cols_to_keep = ['BMI', 'CIG_0', 'DBWT',
                'DOB_MM', 'DOB_TT', 'DOB_WK', 'DWgt_R',
                'FAGECOMB', 'MAGER', 'M_Ht_In', 'NO_INFEC',
                'NO_RISKS', 'PRIORLIVE', 'PREVIS', 'PWgt_R', 'SEX', 'WTGAIN']

df = pandas.read_csv("US_births(2018).csv", delimiter=',', usecols=cols_to_keep, dtype=None)
df.loc[df['SEX'] == 'M', 'SEX'] = 0
df.loc[df['SEX'] == 'F', 'SEX'] = 1

q_min = df.quantile(0.03)
q_max = df.quantile(0.97)
df = df[~((df < q_min) | (df > q_max)).any(axis=1)]

# plt.title('Relation between mother BMI and child weight', fontsize=14)
# plt.xlabel('BMI', fontsize=14)
# plt.ylabel('Child weight', fontsize=14)

sample = df.sample(5000, random_state=10)
plt.scatter(sample['BMI'], sample['DBWT'], color='red', s=5)
plt.grid(True)
plt.show()

# plt.style.use('ggplot')
# df.DBWT.plot(kind='hist', color='purple', edgecolor='black')
# plt.title('Distribution of child weight', size=24)
# plt.xlabel('Child weight (g)', size=18)
# plt.ylabel('Occurrences', size=18)
# plt.show()

# sample2 = df.sample(100000)
# x = sample2['BMI']
# y = sample2['DBWT']
# plt.scatter(x, y, color='red', s=4, alpha=.02)
# m = np.polyfit(x, y, 1)
# plt.plot(x, m[0]*x + m[1])
# plt.grid(True)
# plt.show()