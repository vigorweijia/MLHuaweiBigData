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
train_gps_path = 'event_port/train0523.csv'
test_data_path = 'event_port/A_testData0531.csv'
order_data_path = 'event_port/loadingOrderEvent.csv'
port_data_path = 'event_port/port.csv'


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

def get_data(data, mode='train'):
    assert mode == 'train' or mode == 'test'
    #类型转换
    if mode == 'train':
        data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True)
    elif mode == 'test':
        #data['temp_timestamp'] = data['timestamp']
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['loadingOrder'] = data['loadingOrder'].astype(str)
    data['latitude'] = data['latitude'].astype(float)
    data['speed'] = data['speed'].astype(float)
    data['direction'] = data['direction'].astype(float)
    return data


def mean_skip_zero(arr):
    number=0
    mysum=0
    for i,v in arr.iteritems():
        if v!=0:
            number+=1
            mysum+=v
    if number==0:
        return 0
    else:
        mysum/number
def MY_MSE_skip_zero(arr):
    number = 0
    mysum = 0
    for i, v in arr.iteritems():
        if v != 0:
            number += 1
            mysum += v
    if number == 0:
        average=0
    else:
        average=mysum / number
    res=0
    for i,v in arr.iteritems():
        if v!=0:
            res+=np.square(average-v)
    if number==0:
        return 0
    else:
        return res/number
def get_time(arr):
    return (arr.max()-arr.min()).total_seconds()

#下面这些表示该函数返回后的结果中的列名，其中的loadingOrder，label，count不是训练特征，其他的都是特征
#loadingOrder,distance,mean_speed,speed_mse,mean_speed_skip0,speed_mse_skip0
#anchor_0_6,anchor_7_15,label(时间、label),count,anchor_ratio_0_6,anchor_ratio_7_15,
# first_longtitude,first_latitude,last_longtitude,last_latitude
def get_feature_train(df):
    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    #首先按照订单号进行排序，然后按照时间进行排序
    df['lat_diff']=df.groupby('loadingOrder')['latitude'].diff(1)#计算相邻两个时间点上的经纬度差
    df['lon_diff']=df.groupby('loadingOrder')['longitude'].diff(1)
    df['point_to_point']=df.apply(lambda x:geodistance(x['latitude'],x['longitude'],x['latitude']-
    x['lat_diff'],x['longitude']-x['lon_diff']) if True else 0,axis=1) #计算当前这一点与上一点之间的距离
    dis=df.groupby('loadingOrder')['point_to_point'].agg('sum').reset_index()#dis表示每一个订单中对应的总距离
    dis.columns=['loadingOrder','distance']
    mean_speed=df.groupby('loadingOrder')['speed'].agg(['mean','var',mean_skip_zero,MY_MSE_skip_zero]).reset_index()#求出速度，速度的方差，有0，无0的
    mean_speed.columns=['loadingOrder','mean_speed','speed_mse','mean_speed_skip0','speed_mse_skip0']
    df['anchor_0_6']=df.apply(lambda x: 1 if x['speed']<=6 else 0,axis=1)#抛锚次数
    df['anchor_7_15']=df.apply(lambda x:1 if x['speed']>6 and x['speed']<=15 else 0,axis=1)
    res_df=df.groupby('loadingOrder').agg({'anchor_0_6':['sum'],'anchor_7_15':['sum']}).reset_index()
    res_df.columns=['loadingOrder','anchor_0_6','anchor_7_15']#hhhhhhhhhhhh
    a=df.groupby('loadingOrder')['timestamp'].agg(['count',get_time]).reset_index()
    a.columns=('loadingOrder','count','label')
    res_df=res_df.merge(a,on='loadingOrder')
    #res_df['label']=df.groupby('loadingOrder')['timestamp'].agg(get_time).reset_index()#时间
    res_df['anchor_ratio_0_6']=res_df['anchor_0_6']/res_df['count']
    res_df['anchor_ratio_7_15']=res_df['anchor_7_15']/res_df['count']
    res_df=res_df.merge(dis,on='loadingOrder')
    res_df=res_df.merge(mean_speed,on='loadingOrder')
    first_df = df.sort_values('timestamp').groupby('loadingOrder', as_index=False).first()  # 找出最近的时间戳
    first_df = first_df[['loadingOrder', 'longitude', 'latitude']]
    first_df.columns = ['loadingOrder', 'first_longitude', 'first_latitude']
    last_df = df.sort_values('timestamp', ascending=False).groupby('loadingOrder', as_index=False).first()
    last_df = last_df[['loadingOrder', 'longitude', 'latitude']]
    last_df.columns = ['loadingOrder', 'last_longitude', 'last_latitude']
    first_df = first_df.merge(last_df, on='loadingOrder')  # 存储的是第一个经纬度和最后一个经纬度
    res_df = res_df.merge(first_df, on='loadingOrder')
    res_df.reset_index(drop=True)
    #应该把count这一列删去？，count是GPS的检测次数
    return res_df

#loadingOrder,distance,mean_speed,speed_mse,mean_speed_skip0,speed_mse_skip0
#anchor_0_6,anchor_7_15,label(时间、label),count,anchor_ratio_0_6,anchor_ratio_7_15,
# first_longitude,first_latitude,last_longitude,last_latitude
def get_feature_test(df,port_data_path):
    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    # 首先按照订单号进行排序，然后按照时间进行排序
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)  # 计算相邻两个时间点上的经纬度差
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['point_to_point'] = df.apply(lambda x: geodistance(x['latitude'], x['longitude'], x['latitude'] -
    x['lat_diff'], x['longitude'] - x['lon_diff']) if True else 0,axis=1)  # 计算当前这一点与上一点之间的距离
    dis = df.groupby('loadingOrder')['point_to_point'].agg('sum').reset_index()  # dis表示每一个订单中对应的总距离
    dis.columns=['loadingOrder','previous_dis']
    #接下来计算后半段的距离
    back_dis=df.sort_values('timestamp',ascending=False).groupby('loadingOrder',as_index=False).first()#找出最远的那个时间戳
    back_dis['dest']=back_dis.apply(lambda x:x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-')+1:] if True else '',axis=1)#提取出终点港口
    ports=pd.read_csv(port_data_path)#读取港口文件
    ports['LONGITUDE']=ports['LONGITUDE'].astype(float)
    ports['LATITUDE']=ports['LATITUDE'].astype(float)
    dict_ports={}#存到一个字典里
    for index,row in ports.iterrows():
        dict_ports[row['TRANS_NODE_NAME']]=(row['LONGITUDE'],row['LATITUDE'])#港口名是key，经纬度是value
    #已经获得了终点港口的经纬度，接下来可以计算距离
    back_dis['dest_lon']=back_dis.apply(lambda x:dict_ports[x['dest']][0],axis=1)
    back_dis['dest_lat']=back_dis.apply(lambda  x:dict_ports[x['dest']][1],axis=1)
    back_dis['back_dis']=back_dis.apply(lambda x:geodistance(x['longitude'],x['latitude'],x['dest_lon'],x['dest_lat']) if True else 0,axis=1)
    temp=back_dis[['loadingOrder','back_dis']]
    dis=dis.merge(temp,on='loadingOrder')
    dis['distance']=dis['back_dis']+dis['previous_dis']
    #dis['distance']=dis.apply(lambda x:dis['back_dis']+dis['previous_dis'] if True else 0,axis=1)#dis中的列名有loadingOrder,previous_dis,back_dis,distance
    dis=dis[['loadingOrder','distance']]

    mean_speed = df.groupby('loadingOrder')['speed'].agg(
        ['mean', 'var', mean_skip_zero, MY_MSE_skip_zero]).reset_index()  # 求出速度，速度的方差，有0，无0的
    mean_speed.columns = ['loadingOrder', 'mean_speed', 'speed_mse', 'mean_speed_skip0', 'speed_mse_skip0']
    df['anchor_0_6'] = df.apply(lambda x: 1 if x['speed'] <= 6 else 0, axis=1)  # 抛锚次数
    df['anchor_7_15'] = df.apply(lambda x: 1 if x['speed'] > 6 and x['speed'] <= 15 else 0, axis=1)
    res_df = df.groupby('loadingOrder').agg({'anchor_0_6': ['sum'], 'anchor_7_15': ['sum']}).reset_index()
    res_df.columns = ['loadingOrder', 'anchor_0_6', 'anchor_7_15']  # hhhhhhhhhhhh
    a = df.groupby('loadingOrder')['timestamp'].agg(['count', get_time]).reset_index()
    a.columns = ('loadingOrder', 'count', 'label')
    res_df = res_df.merge(a, on='loadingOrder')
    # res_df['label']=df.groupby('loadingOrder')['timestamp'].agg(get_time).reset_index()#时间
    res_df['anchor_ratio_0_6'] = res_df['anchor_0_6'] / res_df['count']
    res_df['anchor_ratio_7_15'] = res_df['anchor_7_15'] / res_df['count']
    res_df = res_df.merge(dis, on='loadingOrder')
    res_df = res_df.merge(mean_speed, on='loadingOrder')
    #back_dis=df.sort_values('timestamp',ascending=False).groupby('loadingOrder',as_index=False).first()#找出最远的那个时间戳
    first_df=df.sort_values('timestamp').groupby('loadingOrder',as_index=False).first()#找出最近的时间戳
    first_df=first_df[['loadingOrder','longitude','latitude']]
    first_df.columns=['loadingOrder','first_longitude','first_latitude']
    last_df=df.sort_values('timestamp',ascending=False).groupby('loadingOrder',as_index=False).first()
    last_df=last_df[['loadingOrder','longitude','latitude']]
    last_df.columns=['loadingOrder','last_longitude','last_latitude']
    first_df=first_df.merge(last_df,on='loadingOrder')#存储的是第一个经纬度和最后一个经纬度
    res_df=res_df.merge(first_df,on='loadingOrder')
    res_df.reset_index(drop=True)
    # 应该把count这一列删去？，count是GPS的检测次数
    return res_df


def mse_score_eval(preds, valid):
    labels = valid.get_label()
    scores = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse_score', scores, True


def build_model(train, test, pred, label, seed=1080, is_shuffle=True):
    train_pred = np.zeros((train.shape[0],))
    test_pred = np.zeros((test.shape[0],))
    n_splits = 10
    # Kfold
    fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
    kf_way = fold.split(train[pred])
    # params
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'num_leaves': 36,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.7,
        'bagging_freq': 6,
        'seed': 8,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': 8,
        'verbose': 1,
    }
    # train
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train[pred].iloc[train_idx], train[label].iloc[train_idx]
        valid_x, valid_y = train[pred].iloc[valid_idx], train[label].iloc[valid_idx]
        # 数据加载
        n_train = lgb.Dataset(train_x, label=train_y)
        n_valid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=n_train,
            num_boost_round=3000,
            valid_sets=[n_valid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=mse_score_eval
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test[pred], num_iteration=clf.best_iteration) / fold.n_splits

    test['label'] = test_pred

    return test[['loadingOrder', 'label']]


def main():
    train_data = pd.read_csv(train_gps_path, nrows=1000, header=None)
    train_data.columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
                          'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                          'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']

    test_data = pd.read_csv(test_data_path)
    #print(test_data.columns)

    train_data = get_data(train_data, mode='train')
    test_data = get_data(test_data, mode='test')

    train = get_feature_train(train_data)
    test = get_feature_test(test_data, port_data_path)
    #print(train.columns)
    #print(test.columns)
    features = [c for c in train.columns if c not in ['count', 'label', 'loadingOrder']]
    train.to_csv('train.csv')
    test.to_csv('test.csv')
    print('FEATURES:'+str(features))

    train['anchor_0_6'] = train['anchor_0_6'].astype(float)
    train['anchor_7_15'] = train['anchor_7_15'].astype(float)
    train['anchor_ratio_0_6'] = train['anchor_ratio_0_6'].astype(float)
    train['anchor_ratio_7_15'] = train['anchor_ratio_7_15'].astype(float)
    train['distance'] = train['distance'].astype(float)
    train['mean_speed'] = train['mean_speed'].astype(float)
    train['speed_mse'] = train['speed_mse'].astype(float)
    train['mean_speed_skip0'] = train['mean_speed_skip0'].astype(float)
    train['speed_mse_skip0'] = train['speed_mse_skip0'].astype(float)
    train['first_longitude']=train['first_longitude'].astype(float)
    train['first_latitude']=train['first_latitude'].astype(float)
    train['last_longitude'] = train['last_longitude'].astype(float)
    train['last_latitude'] = train['last_latitude'].astype(float)

    test['anchor_0_6'] = test['anchor_0_6'].astype(float)
    test['anchor_7_15'] = test['anchor_7_15'].astype(float)
    #train['count'] = train['count'].astype(float)
    #train['label'] = train['label'].astype(float)
    test['anchor_ratio_0_6'] = test['anchor_ratio_0_6'].astype(float)
    test['anchor_ratio_7_15'] = test['anchor_ratio_7_15'].astype(float)
    test['distance'] = test['distance'].astype(float)
    test['mean_speed'] = test['mean_speed'].astype(float)
    test['speed_mse'] = test['speed_mse'].astype(float)
    test['mean_speed_skip0'] = test['mean_speed_skip0'].astype(float)
    test['speed_mse_skip0'] = test['speed_mse_skip0'].astype(float)
    test['first_longitude'] = test['first_longitude'].astype(float)
    test['first_latitude'] = test['first_latitude'].astype(float)
    test['last_longitude'] = test['last_longitude'].astype(float)
    test['last_latitude'] = test['last_latitude'].astype(float)

    result = build_model(train, test, features, 'label', is_shuffle=True)
    result.to_csv('result.csv')
    #构建并训练模型，result就是预测出的消耗的时间，再加上起始时间就是ETA；
    test_data = test_data.merge(result, on='loadingOrder', how='left')
    test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    test_data['timestamp'] = test_data['timestamp']
    # 整理columns顺序
    result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]
    result.to_csv('testout.csv')


if __name__ == '__main__':
    main()