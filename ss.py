import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

data_set = pd.read_csv('US_births(2018).csv', sep=',',
                       usecols=[2, 3, 13, 18, 25, 35, 40, 43, 44, 45, 46, 49, 52])
data_frame_temp = pd.DataFrame(data_set)

# Wstępnie zmieniłem też typy danych, które przechowuje data frame ze względu na dość dużą liczbę danych. W ten
# sposób ograniczę ilość pamięci potrzebnej do działania na zbiorze.
data_frame = data_frame_temp[data_frame_temp.columns[1:len(data_frame_temp.columns)]].astype(np.uint8)
data_frame.insert(0, 'BMI', data_frame_temp['BMI'].astype(np.float32))

# Zanim skorzystam z metod uczenia maszynowego, przygotuję dane w celu pozbycia się rekordów
# z brakującymi informacjami. W dokumentacji zbioru można sprawdzić, że brak danych praktycznie zawsze
# oznacza się za pomocą liczby 9 lub 99. Korzystając z tej informacji, usunę właśnie te rekordy, które zawierą
# wyżej wymienione liczby. Oczywiście sprawdziłem wcześniej czy mogę to zrobić bez utraty ważnych rekordów.
for column in range(0, len(data_frame.columns)):
    data_frame = data_frame[data_frame[data_frame.columns[column]] != 99]
for column in [3, 6, 11]:
    data_frame = data_frame[data_frame[data_frame.columns[column]] != 9]

#print(data_frame[data_frame.columns[3]])
# for i in range(len(data_frame.columns)):
#     print(data_frame[data_frame.columns[i]])
pd.set_option('display.max_columns', None)
print(data_frame.head())



#plt.style.use('ggplot')
smpl = df1.sample(10000)
sc = plt.scatter(x=smpl['BMI'], y=smpl['M_Ht_In'], c=smpl['DBWT'], cmap=plt.get_cmap("jet"), alpha=.25)
plt.title('BMI a wysokość matki', size=24)
plt.xlabel('BMI', size=18)
plt.ylabel('M_Ht_In', size=18)
plt.colorbar(sc)
plt.show()