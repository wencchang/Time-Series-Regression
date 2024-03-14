#%% 載入套件
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
#%% 引入資料
raw = pd.read_excel('新竹_2021.xls')
raw.info()
raw
#%% 欄名重命名
hours = raw.columns[3:].tolist()
raw.columns = ['Site', 'Date', 'Target'] + hours
raw.columns
#%% 去掉Site & 分隔線
raw = raw.drop(['Site'], axis=1).drop(0)
raw
#%% 取出10.11.12月資料
raw.Date = pd.to_datetime(raw.Date)
raw = raw[raw.Date > '2021-09-30']
raw = raw.reset_index(drop=True)
raw
#%% 去掉字串後多餘的空白並依類別分組
raw['Target'] = raw['Target'].apply(lambda x: x.strip())
sector = raw.groupby('Target')
sector.size()
#%% 欄名為類別欄位內容為二維轉一維數據
df_all = pd.DataFrame()
categorys = raw['Target'].unique()
for category in categorys:
    temp = sector.get_group(category).loc[:, 0:]
    df_all[category] = temp.values.reshape(-1)
df_all
#%% 缺失值以及無效值以前一小時
record_front = []

for col in df_all:
    
    record = []
    
    for data in df_all[col]:
        if type(data) != str:
            num = data
            break
    
    for data in df_all[col]:
        if type(data) != str:
            num = data
        # else:
            # print(col, data)
        
        if 'NR' in str(data):
            record.append(0) 
        else:
            record.append(num)
    
    record_front.append(record)
#%% 缺失值以及無效值以後一小時
record_back = []

for col in df_all:
    
    record = []
    
    reverse = list(reversed(df_all[col]))
    for data in reverse:
        if type(data) != str:
            num = data
            break
    
    for data in reverse:
        if type(data) != str:
            num = data
        
        if 'NR' in str(data):
            record.append(0)
        else:
            record.append(num)
    
    record = list(reversed(record))
    record_back.append(record)
#%% 取平均
rows = df_all.shape[0]
cols = df_all.shape[1]

for col in range(cols):
    for row in range(rows):
        df_all.iloc[row, col] = (record_front[col][row] + record_back[col][row])/2
 #%% 將資料切割成訓練集(10.11月)以及測試集(12月)
df_train = df_all[:1464]
df_test = df_all[1464:]
#%% 將資料形式轉換為行(row)代表18種屬性，欄(column)代表逐時數據資料。將訓練集每18行合併，轉換成維度為(18,61*24)的DataFrame(每個屬性都有61天*24小時共1464筆資料)
df_train = df_train.T
df_test = df_test.T
df_train

#%% 第0~5小時
df_train.iloc[:, 0:6]
#%% 第6小時(未來第一小時)的PM2.5值
df_train[ df_train.index == 'PM2.5' ].iloc[:, 6]
#%% 取6小時為一單位切割，例如第一筆資料為第0~5小時的資料(X[0])，下一筆資料為第1~6小時的資料(X[1])
def cut_x(dataset, interval):
    dataset_cut = []
    cols = dataset.shape[1]
    for i in range(cols):
        if i + interval < cols:
            temp = dataset.iloc[:, i:i+interval]
            dataset_cut.append( temp.values.reshape(-1) )
    return pd.DataFrame(dataset_cut)
#%% 預測第6小時(未來第一小時)的PM2.5值(Y[0])，下一筆資料預測第12小時的PM2.5值(Y[1])
def cut_y(dataset, interval, num):
    dataset_cut = []
    cols = dataset.shape[1]
    for i in range(cols):
        if i + (interval+num-1) < cols:
            temp = dataset.iloc[:, i+(interval+num-1)]
            dataset_cut.append(temp.values)
    return pd.DataFrame(dataset_cut)

#%% X請分別取只有PM2.5和所有18種屬性，預測目標分未來第一個小時和未來第六個小時

x_train_all = cut_x(df_train, 6)
x_train_pm25 = cut_x(df_train[df_train.index == 'PM2.5'], 6)
y_train_one = cut_y(df_train[df_train.index == 'PM2.5'], 6, 1)
y_train_six = cut_y(df_train[df_train.index == 'PM2.5'], 6, 6)

x_test_all = cut_x(df_test, 6)
x_test_pm25 = cut_x(df_test[df_test.index == 'PM2.5'], 6)
y_test_one = cut_y(df_test[df_test.index == 'PM2.5'], 6, 1)
y_test_six = cut_y(df_test[df_test.index == 'PM2.5'], 6, 6)

print(len(x_train_all))
print(len(x_train_pm25))
print(len(y_train_one))
print(len(y_train_six))
#%% LinearRegression
reg = LinearRegression()
#%% LR: 所有屬性 & 未來第一個小時
reg.fit(x_train_all, y_train_one)
pred = reg.predict(x_test_all)
mean_absolute_error(y_test_one, pred)
print('LR,所有屬性,未來第一個小時:',mean_absolute_error(y_test_one, pred))
#%% LR: PM2.5 & 未來第一個小時
reg.fit(x_train_pm25, y_train_one)
pred = reg.predict(x_test_pm25)
mean_absolute_error(y_test_one, pred)
print('LR,PM2.5,未來第一個小時:',mean_absolute_error(y_test_one, pred))
#%% LR: 所有屬性 & 未來第六個小時
reg.fit(x_train_all[:-5], y_train_six)
pred = reg.predict(x_test_all[:-5])
mean_absolute_error(y_test_six, pred)
print('LR,所有屬性,未來第六個小時:',mean_absolute_error(y_test_six, pred))
#%% LR: PM2.5 & 未來第六個小時
reg.fit(x_train_pm25[:-5], y_train_six)
pred = reg.predict(x_test_pm25[:-5])
mean_absolute_error(y_test_six, pred)
print('LR,PM2.5,未來第六個小時:',mean_absolute_error(y_test_six, pred))

#%% XGBRegressor
xgb = XGBRegressor()
#%% XGB: 所有屬性 & 未來第一個小時
xgb.fit(x_train_all, y_train_one)
pred = xgb.predict(x_test_all)
mean_absolute_error(y_test_one, pred)
print('XGB,所有屬性,未來第一個小時:',mean_absolute_error(y_test_one, pred))
#%% XGB: PM2.5 & 未來第一個小時
xgb.fit(x_train_pm25, y_train_one)
pred = xgb.predict(x_test_pm25)
mean_absolute_error(y_test_one, pred)
print('XGB,PM2.5,未來第一個小時:',mean_absolute_error(y_test_one, pred))

#%% XGB: 所有屬性 & 未來第六個小時
xgb.fit(x_train_all[:-5], y_train_six)
pred = xgb.predict(x_test_all[:-5])
mean_absolute_error(y_test_six, pred)
print('XGB,所有屬性,未來第六個小時:',mean_absolute_error(y_test_six, pred))

#%% XGB: PM2.5 & 未來第六個小時
xgb.fit(x_train_pm25[:-5], y_train_six)
pred = xgb.predict(x_test_pm25[:-5])
mean_absolute_error(y_test_six, pred)
print('XGB,PM2.5,未來第六個小時:',mean_absolute_error(y_test_six, pred))

