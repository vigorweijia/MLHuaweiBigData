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

def Del_Wrong_Diff():
    for i in range(15,30):
        chunkPath = "wxyWash/chunk_"+str(i)+".csv"
        chunk = pd.read_csv(chunkPath,header=None)
        chunk.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                         'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                         'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        chunk.drop([0],inplace=True)
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
        chunk['mark']=chunk.apply(lambda x:1 if x['TRANSPORT_TRACE'].find('-')==x['TRANSPORT_TRACE'].rfind('-') else 0,axis=1)
        #相等的话说明是只有1个'-'，此时是1，没有中转，保留这样的
        chunk=chunk[chunk['mark']==1]
        chunk=chunk[['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                        'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                        'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']]
        outputFilePath = "sshWash/chunk_"+str(i)+".csv"
        print(outputFilePath)
        chunk.to_csv(outputFilePath,index=False)

if __name__=='__main__':
    Del_Wrong_Diff()