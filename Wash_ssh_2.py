import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb
#lightgbm简单理解就是一种很好用的决策树集成算法；
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')
# baseline只用到gps定位数据，即train_gps_path
EARTH_REDIUS = 6378.137
train_gps_path = 'D:/data1/washed_train_data.csv'
test_data_path = 'D:/A_testData0531.csv'
order_data_path = 'D:/event_port/loadingOrderEvent.csv'
port_data_path = 'D:/event_port/port.csv'
def geodistance(lng1,lat1,lng2,lat2):
    #print(type(lng1),type(lng2),type(lat1),type(lat2))
    #print(lng1,lng2,lat1,lat2)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance


def Del_still_Order():
    chunk = pd.read_csv('merge/merged_train_data.csv')#首先进行类型转换
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], infer_datetime_format=True)
    chunk['longitude'] = chunk['longitude'].astype(float)
    chunk['loadingOrder'] = chunk['loadingOrder'].astype(str)
    chunk['latitude'] = chunk['latitude'].astype(float)
    chunk['speed'] = chunk['speed'].astype(float)
    chunk['direction'] = chunk['direction'].astype(float)
    chunk.sort_values(['loadingOrder','timestamp'],inplace=True)
    print(chunk.shape)
    a=chunk.groupby('loadingOrder').agg('mean').reset_index()
    a=a[['loadingOrder']]
    a.to_csv('res1.csv',index=False)
    print(a.shape)
    #先排序；
    first_df = chunk.sort_values('timestamp').groupby('loadingOrder', as_index=False).first()  # 找出最近的时间戳
    first_df = first_df[['loadingOrder', 'longitude', 'latitude']]
    first_df.columns = ['loadingOrder', 'first_longitude', 'first_latitude']
    last_df = chunk.sort_values('timestamp', ascending=False).groupby('loadingOrder', as_index=False).first()
    last_df = last_df[['loadingOrder', 'longitude', 'latitude']]
    last_df.columns = ['loadingOrder', 'last_longitude', 'last_latitude']
    first_df = first_df.merge(last_df, on='loadingOrder')  # 存储的是第一个经纬度和最后一个经纬度
    first_df['dis']=first_df.apply(lambda x:geodistance(x['first_longitude'],x['first_latitude'],x['last_longitude'],x['last_latitude']),axis=1)
    first_df=first_df[['loadingOrder','dis']]
    #first_df['dis']=first_df[first_df['dis']<=10]
    chunk=chunk.merge(first_df,on='loadingOrder')
    chunk=chunk[chunk['dis']>10]#只保留dis比较大的；
    chunk = chunk[['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                   'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                   'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']]
    print(chunk.shape)
    b = chunk.groupby('loadingOrder').agg('mean').reset_index()
    b = b[['loadingOrder']]
    b.to_csv('res2.csv', index=False)
    print(b.shape)
    chunk.to_csv('sshWash_2/del_still_order_data.csv',index=False)

if __name__=='__main__':
    Del_still_Order()