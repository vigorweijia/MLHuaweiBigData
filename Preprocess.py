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
train_gps_path = 'D:/train0523/chunck_1.csv'
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
def Wash():
    CHUNKSIZE = 5000000
    NROWS = CHUNKSIZE * 10
    data = pd.read_csv(train_gps_path, header=None, chunksize=CHUNKSIZE)


    for i,chunk in enumerate(data):
        chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                        'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                        'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        chunk = chunk.astype(str)
        chunk = chunk[chunk['TRANSPORT_TRACE'].str.contains('-')]
        filePath = 'washed/chunck_'+str(i)+'.csv'
        print('generate '+filePath+'...')
        chunk.to_csv(filePath)


def Del_Wrong_Diff():
    chunkPath = 'merge/merged_train_data.csv'
    chunk = pd.read_csv(chunkPath)
    del chunk['Unnamed: 0']
    chunk.columns = ['a','loadingOrder', 'carrierName', 'timestamp', 'longitude',
                    'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                    'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], infer_datetime_format=True)
    chunk['longitude'] = chunk['longitude'].astype(float)
    chunk['loadingOrder'] = chunk['loadingOrder'].astype(str)
    chunk['latitude'] = chunk['latitude'].astype(float)
    chunk['speed'] = chunk['speed'].astype(float)
    chunk['direction'] = chunk['direction'].astype(float)
    chunk.sort_values(['loadingOrder','timestamp'],inplace=True)
    chunk['time_diff'] = chunk.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()#
    chunk['overlap'] = chunk.apply(lambda x: 1 if str(x['time_diff']) != 'nan' and x['time_diff'] <= 30 else 0,axis=1)  # time_diff小于30秒且time_diff不为nan的话就去掉
    chunk = chunk[chunk['overlap'] != 1]  # 看做是重复的记录，去掉
    #print('hhh')
    #for i in range(5):
    chunk['lat_diff']=chunk.groupby('loadingOrder')['latitude'].diff(1)
    chunk['lon_diff']=chunk.groupby('loadingOrder')['longitude'].diff(1)
    chunk['time_diff']=chunk.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    chunk['lat_diff_'] = chunk.groupby('loadingOrder')['latitude'].diff(-1)
    chunk['lon_diff_'] = chunk.groupby('loadingOrder')['longitude'].diff(-1)
    chunk['time_diff_'] =-chunk.groupby('loadingOrder')['timestamp'].diff(-1).dt.total_seconds()
    chunk['point_to_point'] = chunk.apply(lambda x: geodistance(x['latitude'], x['longitude'], x['latitude'] - x['lat_diff'],x['longitude'] - x['lon_diff']), axis=1)  # 计算距离
    chunk['average_speed'] = chunk.apply(lambda x: x['point_to_point'] / x['time_diff'] * 3600 if str(x['time_diff']) != 'nan'else 0, axis=1)  # 计算的是km/h
    chunk['point_to_point_']=chunk.apply(lambda x:geodistance(x['latitude'], x['longitude'], x['latitude'] -x['lat_diff_'], x['longitude'] - x['lon_diff_']) if True else 0,axis=1)#计算距离
    chunk['average_speed_']=chunk.apply(lambda x: x['point_to_point_']/x['time_diff_']*3600 if str(x['time_diff_'])!='nan'else 0,axis=1)#计算的是km/h
    chunk['mark']=chunk.apply(lambda x:1 if x['average_speed']<50 and x['average_speed_']<50 else 0,axis=1)
    chunk=chunk[chunk['mark']==1]
    #print('kkk')
    filePath = 'sshWash/washed_train_data.csv'
    chunk=chunk[['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                    'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                    'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']]
    chunk.to_csv(filePath)


if __name__=='__main__':
    Del_Wrong_Diff()