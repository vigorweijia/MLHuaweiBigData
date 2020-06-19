import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
#lightgbm简单理解就是一种很好用的决策树集成算法；
from math import radians, cos, sin, asin, sqrt
import warnings
warnings.filterwarnings('ignore')
# baseline只用到gps定位数据，即train_gps_path
EARTH_REDIUS = 6378.137
#train_gps_path = 'event_port/train0523.csv'
#test_data_path = 'event_port/A_testData0531.csv'
#order_data_path = 'event_port/loadingOrderEvent.csv'
#port_data_path = 'event_port/port.csv'
train_gps_path = 'new_train_data.csv'
test_data_path = 'event_port/A_testData0531.csv'
order_data_path = 'event_port/loadingOrderEvent.csv'
port_data_path = 'event_port/port_archived.csv'

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
        data['temp_timestamp'] = data['timestamp']
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
        #print(0)
        return 0
    else:
        #print(mysum/number)
        return mysum/number
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
    mean_speed=df.groupby('loadingOrder')['speed'].agg(['mean','var','median','max','min',mean_skip_zero,MY_MSE_skip_zero]).reset_index()#求出速度，速度的方差，有0，无0的
    mean_speed.columns=['loadingOrder','mean_speed','speed_mse','median_speed','max_speed','min_speed','mean_speed_skip0','speed_mse_skip0']
    #print(mean_speed)
    #print('hhh')
    df['anchor_0_6']=df.apply(lambda x: 1 if x['speed']<=6 else 0,axis=1)#抛锚次数
    df['anchor_7_15']=df.apply(lambda x:1 if x['speed']>6 and x['speed']<=15 else 0,axis=1)
    res_df=df.groupby('loadingOrder').agg({'anchor_0_6':['sum'],'anchor_7_15':['sum']}).reset_index()
    res_df.columns=['loadingOrder','anchor_0_6','anchor_7_15']#hhhhhhhhhhhh
    a = df.groupby('loadingOrder')['timestamp'].agg(['count', 'max', 'min']).reset_index()
    a.columns = ('loadingOrder', 'count', 'max', 'min')
    # a['label']=a.apply(lambda x:(x['max']-x['min']).total_sconds())
    a['label'] = (a['max'] - a['min']).dt.total_seconds()
    a = a[['loadingOrder', 'count', 'label']]
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
'''
def test_train_dis(x,train_data,df):
    train_data=train_data[train_data['loadingOrder']==x['train_loadingOrder']]#挑出那些是匹配的
    df=df[df['loadingOrder']==x['test_loadingOrder']]#挑出对应的
    train_data=train_data.reset_index()
    df=df.reset_index()
    total=0
    for index,row in df.iterrows():#对于test轨迹中的每一个点，求出该点到当前train轨迹中的最小距离
        train_data['P2P']=train_data.apply(lambda x:geodistance(row['longitude'],row['latitude'],x['longitude'],x['latitude']),axis=1)#计算test中的一个点到train_data所有点的距离
        #存入P2P这一列中
        temp=train_data.groupby('loadingOrder')['P2P'].agg('min').reset_index()
        #找最小的P2P
        #print(temp.shape)
        total+=temp.iloc[0,1]#加入结果中
        #print(temp.iloc[0,1])
    print('wxycgp')
    return total/df.shape[0]
'''
#loadingOrder,distance,mean_speed,speed_mse,mean_speed_skip0,speed_mse_skip0
#anchor_0_6,anchor_7_15,label(时间、label),count,anchor_ratio_0_6,anchor_ratio_7_15,
# first_longitude,first_latitude,last_longitude,last_latitude


def match_order(x,temp1):
    for index,row in temp1.iterrows():
        if x['arrival']==row['arrival'] and x['departure']==row['departure']:
            return row['distance']
    for index,row in temp1.iterrows():
        if x['arrival']==row['departure'] and x['departure']==row['arrival']:
            return row['distance']
    return x['linear_distance']


def get_feature_test(df,port_data_path,train,train_data):
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
    dis['linear_distance']=dis['back_dis']+dis['previous_dis']
    #dis['distance']=dis.apply(lambda x:dis['back_dis']+dis['previous_dis'] if True else 0,axis=1)#dis中的列名有loadingOrder,previous_dis,back_dis,distance
    dis=dis[['loadingOrder','linear_distance']]
    #上面这一段是计算距离的


    mean_speed = df.groupby('loadingOrder')['speed'].agg(['mean', 'var', 'median','max','min',mean_skip_zero, MY_MSE_skip_zero]).reset_index()  # 求出速度，速度的方差，有0，无0的
    mean_speed.columns = ['loadingOrder', 'mean_speed','median_speed', 'max_speed','min_speed','speed_mse', 'mean_speed_skip0', 'speed_mse_skip0']
    df['anchor_0_6'] = df.apply(lambda x: 1 if x['speed'] <= 6 else 0, axis=1)  # 抛锚次数
    df['anchor_7_15'] = df.apply(lambda x: 1 if x['speed'] > 6 and x['speed'] <= 15 else 0, axis=1)
    res_df = df.groupby('loadingOrder').agg({'anchor_0_6': ['sum'], 'anchor_7_15': ['sum']}).reset_index()
    res_df.columns = ['loadingOrder', 'anchor_0_6', 'anchor_7_15']  # hhhhhhhhhhhh
    a = df.groupby('loadingOrder')['timestamp'].agg(['count', 'max', 'min']).reset_index()
    a.columns = ('loadingOrder', 'count', 'max', 'min')
    # a['label']=a.apply(lambda x:(x['max']-x['min']).total_sconds())
    a['label'] = (a['max'] - a['min']).dt.total_seconds()


    res_df = res_df.merge(a, on='loadingOrder')
    # res_df['label']=df.groupby('loadingOrder')['timestamp'].agg(get_time).reset_index()#时间
    res_df['anchor_ratio_0_6'] = res_df['anchor_0_6'] / res_df['count']
    res_df['anchor_ratio_7_15'] = res_df['anchor_7_15'] / res_df['count']
    res_df = res_df.merge(dis, on='loadingOrder')
    res_df = res_df.merge(mean_speed, on='loadingOrder')
    #back_dis=df.sort_values('timestamp',ascending=False).groupby('loadingOrder',as_index=False).first()#找出最远的那个时间戳
    print('aaa')
    first_df=df.sort_values('timestamp').groupby('loadingOrder',as_index=False).first()#找出最近的时间戳
    first_df=first_df[['loadingOrder','longitude','latitude']]
    first_df.columns=['loadingOrder','first_longitude','first_latitude']
    last_df=df.sort_values('timestamp',ascending=False).groupby('loadingOrder',as_index=False).first()
    last_df=last_df[['loadingOrder','longitude','latitude']]
    last_df.columns=['loadingOrder','last_longitude','last_latitude']
    first_df=first_df.merge(last_df,on='loadingOrder')#存储的是第一个经纬度和最后一个经纬度
    res_df=res_df.merge(first_df,on='loadingOrder')
    print('bbb')
    temp_df = df.groupby('loadingOrder', as_index=False).first()  # 找出测试集中每一个订单的任意一条记录，
    # 之所以可以找任意是因为是为了计算TRACE，任意一条都可以
    temp_df['departure'] = temp_df.apply(lambda x: x['TRANSPORT_TRACE'][0:x['TRANSPORT_TRACE'].find('-')],
                                         axis=1)  # 测试集的起点
    temp_df['arrival'] = temp_df.apply(lambda x: x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-') + 1:],
                                       axis=1)  # 测试集的终点
    temp0 = train_data.groupby('loadingOrder', as_index=False).first()  # 找出训练集中每一个订单的任意一条记录，
    temp0['departure'] = temp0.apply(lambda x: x['TRANSPORT_TRACE'][0:x['TRANSPORT_TRACE'].find('-')],
                                     axis=1)  # 训练集中的起点港口
    temp0['arrival'] = temp0.apply(lambda x: x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-') + 1:],
                                   axis=1)  # 训练集中的终点港口
    ports = pd.read_csv(port_data_path)  # 读取港口文件
    print(ports.columns)
    print('ccc')
    dict_ports = {}  # 存到一个字典里
    for index, row in ports.iterrows():
        dict_ports[row['TRANS_NODE_NAME']] = row['REAL_PORT_NAME']  # 原来的港口名和真实的港口名的映射关系存到字典中
    # 已经获得了终点港口的经纬度，接下来可以计算距离
    temp_df['departure'] = temp_df.apply(lambda x: dict_ports[x['departure']], axis=1)#测试集
    temp_df['arrival'] = temp_df.apply(lambda x: dict_ports[x['arrival']], axis=1)
    temp0['departure'] = temp0.apply(lambda x: dict_ports[x['departure']], axis=1)#训练集
    temp0['arrival'] = temp0.apply(lambda x: dict_ports[x['arrival']], axis=1)  # 分别换成真实的港口名
    #temp_df = temp_df[['loadingOrder', 'departure', 'arrival']]
    temp_df=temp_df.merge(dis,on='loadingOrder')#把直线距离也拼接上temp_df包含了每组的某条订单+departure+arrival+Linear_distance
    #temp_df.columns = ['test_loadingOrder', 'departure', 'arrival']  # 换名
    temp0 = temp0[['loadingOrder', 'departure', 'arrival']]
    #temp1.columns = ['train_loadingOrder', 'departure', 'arrival']
    temp1=temp0.merge(train,on='loadingOrder')#merge的目的是为了提取出train的距离
    temp1=temp1[['loadingOrder', 'departure', 'arrival','distance']]#把训练集的距离提取出来#temp1包含了某一条订单，departure，arrival+
    #temp1.columns=['train_loadingOrder','departure','arrival','distance']#提取出每个训练订单的距离。
    group_train_order=temp1.groupby(['departure','arrival']).agg(['mean','std']).reset_index()
    group_train_order.columns=['departure','arrival','mean','std']
    #new_train_data=temp1.merge(group_train_order,on=['departure','arrival'])
    new_train_data=temp1.merge(train_data,on='loadingOrder')#把匹配的所有订单信息都拿出来
    #group_train_order=group_train_order[group_train_order['std']>=1000]#找出标准差大于1000的那些训练订单
    #b=group_train_order.merge(temp_df,on=['departure','arrival'])#与测试订单做连接
    #print(b.shape)
    #遍历temp_df中的每一行
    dict2={}
    for index,row in group_train_order.iterrows():
        dict2[(row['departure'],row['arrival'])]=(row['mean'],row['std'])

    temp_df['distance']=temp_df.apply(lambda x:match_order(x,dict2,temp0,new_train_data,df),axis=1)
    #temp_df['distance']=temp_df.apply(lambda x:x['distance'] if x['distance']>0 else x['linear_distance'])
    temp_df=temp_df[['loadingOrder','distance']]
    '''
    merge1 = temp1.merge(temp_df, on=['departure', 'arrival'])  # 内连接，不完全一样的就扔掉。#这样
    # merge1中的列有：test_loadingOrder，train_loadingOrder,arrival，departure
    # 将每个test订单都和一个或者多个或者0个train订单对应上
    # 这样的话，一个test订单order可能对应着好几个train订单order，要找到最小的那个。
    print('ddd')
    merge1['similarity'] = merge1.apply(lambda x: test_train_dis(x, train_data, df),
                                        axis=1)  # 这里是计算test_loadingOrder和对应的一个train_loadingOrder轨迹的相似度，
    # 相似度越大，similarity越小
    # 计算出每个test订单和对应的train距离最小的；
    merge1 = merge1.sort_values(['similarity']).groupby('test_loadingOrder', as_index=False).first()  # 找出最小的similarity
    # 找每一组中dis最小的那个;
    print('eee')
    merge1 = merge1[['test_loadingOrder', 'train_loadingOrder']]  # 找出每个test_loadingOrder对应的train_loadingOrder
    merge1.columns = ['test_loadingOrder', 'loadingOrder']
    merge1 = merge1.merge(train, on='loadingOrder')  # 和train_data进行merge的目的是在于获得对应的train订单的距离
    merge1 = merge1[['test_loadingOrder', 'loadingOrder', 'distance']]
    merge1.columns = ['test_loadingOrder', 'train_loadingOrder', 'distance']
    # 有可能有的test_order并没有。
    print('fff')
    mydict = {}
    for index, row in merge1.iterrows():
        mydict[row['test_loadingOrder']] = row['distance']
    # 将test订单和对应的train订单存入到字典中
    res_df['distance'] = res_df.apply(
        lambda x: mydict[x['loadingOrder']] if x['loadingOrder'] in mydict else x['linear_distance'], axis=1)
    # 表示推断出的距离，如果在上面的字典中的话，那么就是存在字典中的距离，否则就是直线距离代替。
    # 这三个一样的
    '''
    res_df=res_df.merge(temp_df,on='loadingOrder')
    #result = res_df[['loadingOrder', 'distance']]
    #result.to_csv('infer_distance.csv')
    print('ggg')
    res_df.reset_index(drop=True)
    # 应该把count这一列删去？，count是GPS的检测次数
    return res_df

def calculate_difference(train_data,train):
    a=train_data.groupby('loadingOrder', as_index=False).first()
    a=a[['loadingOrder','TRANSPORT_TRACE','carrierName']]
    a=a.merge(train,on='loadingOrder')
    a['departure'] = a.apply(lambda x: x['TRANSPORT_TRACE'][0:x['TRANSPORT_TRACE'].find('-')],
                                         axis=1)
    a['arrival'] = a.apply(lambda x: x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-') + 1:],
                                       axis=1)
    ports = pd.read_csv(port_data_path)  # 读取港口文件
    print(ports.columns)
    print('ccc')
    dict_ports = {}  # 存到一个字典里
    for index, row in ports.iterrows():
        dict_ports[row['TRANS_NODE_NAME']] = row['REAL_PORT_NAME']  # 原来的港口名和真实的港口名的映射关系存到字典中
    # 已经获得了终点港口的经纬度，接下来可以计算距离
    a['departure'] = a.apply(lambda x: dict_ports[x['departure']], axis=1)  # 测试集
    a['arrival'] = a.apply(lambda x: dict_ports[x['arrival']], axis=1)
    b=a.groupby(['departure','arrival','carrierName'])['distance'].agg(['count','min','max','median','var','std']).reset_index()
    #c=a.groupby(['departure','arrival','carrierName'])['distance'].agg('count').reset_index()
    b.columns=['departure','arrival','carrierName','count','min','max','median','var','std']
    b.to_csv('group_ports.csv',index=False)

def match_order(x,dict2,temp0,new_train_data,df):
    if (x['departure'],x['arrival']) in dict2:
        if dict2[(x['departure'],x['arrival'])][1]<1000:
            return dict2[(x['departure'],x['arrival'])][0]
        else:#去匹配一个轨迹最相似的
            df = df[df['loadingOrder'] == x['loadingOrder']]#跳出对应的
            temp0=temp0[temp0['departure']==x['departure']]
            temp0=temp0[temp0['arrival']==x['arrival']]
            min=999999999
            res=0
            for index,row in temp0.iterrows():
                a=new_train_data[new_train_data['loadingOrder']==row['loadingOrder']]
                total=0
                for i1,r1 in df.iterrows():
                    a['P2P']=a.apply(lambda x:geodistance(r1['longitude'],r1['latitude'],x['longitude'],x['latitude']),axis=1)
                    hhh=a.groupby('loadingOrder')['P2P'].agg('min').reset_index()
                    total+=hhh.iloc[0,1]
                if total<=min:
                    b=a.groupby('loadingOrder')['distance'].agg('min').reset_index()
                    res=b.iloc[0,1]
        return res
    else:
        return x['linear_distance']




'''
def get_feature_infer_distance(df,port_data_path,train,train_data,res_df):
    #df=pd.read_csv(test_data_path)
    #接下来用另一种方式来计算距离
    #对于每一个test订单，确定其起始港口位置及终止港口，找训练集中的所有订单，假如起始港口和终止港口相同并且承运商相同，那么就首先
    #考虑，然后再考虑起始港口和终止港口相同的，如果没有相同的，再考虑直线距离，或者是找到起始港口以及中止港口最接近的训练订单？
    #将训练订单的距离看成是样本中的距离
    # 上面计算的是直线距离，这里从训练集中找一个相似的距离来计算。
    temp_df = df.sort_values('timestamp').groupby('loadingOrder', as_index=False).first()  # 找出测试集中每一个订单的任意一条记录，
    # 之所以可以找任意是因为是为了计算TRACE，任意一条都可以
    temp_df['departure'] = temp_df.apply(lambda x: x['TRANSPORT_TRACE'][0:x['TRANSPORT_TRACE'].find('-')],
                                         axis=1)  # 测试集的起点
    temp_df['arrival'] = temp_df.apply(lambda x: x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-') + 1:],
                                       axis=1)  # 测试集的终点
    temp1 = train_data.sort_values('timestamp').groupby('loadingOrder', as_index=False).first()  # 找出训练集中每一个订单的任意一条记录，
    temp1['departure'] = temp1.apply(lambda x: x['TRANSPORT_TRACE'][0:x['TRANSPORT_TRACE'].find('-')],
                                     axis=1)  # 训练集中的起点港口
    temp1['arrival'] = temp1.apply(lambda x: x['TRANSPORT_TRACE'][x['TRANSPORT_TRACE'].rfind('-') + 1:],
                                   axis=1)  # 训练集中的终点港口
    ports = pd.read_csv(port_data_path)  # 读取港口文件
    print(ports.columns)
    dict_ports = {}  # 存到一个字典里
    for index, row in ports.iterrows():
        dict_ports[row['TRANS_NODE_NAME']] = row['REAL_PORT_NAME']  # 原来的港口名和真实的港口名的映射关系存到字典中
    # 已经获得了终点港口的经纬度，接下来可以计算距离
    temp_df['departure'] = temp_df.apply(lambda x: dict_ports[x['departure']], axis=1)
    temp_df['arrival'] = temp_df.apply(lambda x: dict_ports[x['arrival']], axis=1)
    temp1['departure'] = temp1.apply(lambda x: dict_ports[x['departure']], axis=1)
    temp1['arrival'] = temp1.apply(lambda x: dict_ports[x['arrival']], axis=1)  # 分别换成真实的港口名
    temp_df = temp_df[['loadingOrder', 'departure', 'arrival']]
    temp_df.columns = ['test_loadingOrder', 'departure', 'arrival']  # 换名
    temp1 = temp1[['loadingOrder', 'departure', 'arrival']]
    temp1.columns = ['train_loadingOrder', 'departure', 'arrival']
    merge1 = temp1.merge(temp_df, on=['departure', 'arrival'])  # 内连接，不完全一样的就扔掉。
    # merge1中的列有：test_loadingOrder，train_loadingOrder,arrival，departure
    # 将每个test订单都和一个或者多个或者0个train订单对应上
    # 这样的话，一个test订单order可能对应着好几个train订单order，要找到最小的那个。
    merge1['similarity'] = merge1.apply(lambda x: test_train_dis(x, train_data, df),
                                        axis=1)  # 这里是计算test_loadingOrder和对应的一个train_loadingOrder轨迹的相似度，
    # 相似度越大，similarity越小
    # 计算出每个test订单和对应的train距离最小的；
    merge1 = merge1.sort_values(['similarity']).groupby('test_loadingOrder', as_index=False).first()  # 找出最小的similarity
    # 找每一组中dis最小的那个;
    merge1 = merge1[['test_loadingOrder', 'train_loadingOrder']]  # 找出每个test_loadingOrder对应的train_loadingOrder
    merge1.columns = ['test_loadingOrder', 'loadingOrder']
    merge1 = merge1.merge(train, on='loadingOrder')  # 和train_data进行merge的目的是在于获得对应的train订单的距离
    merge1 = merge1[['test_loadingOrder', 'loadingOrder', 'distance']]
    merge1.columns = ['test_loadingOrder', 'train_loadingOrder', 'distance']
    # 有可能有的test_order并没有。
    mydict = {}
    for index, row in merge1.iterrows():
        mydict[row['test_loadingOrder']] = row['distance']
    # 将test订单和对应的train订单存入到字典中
    res_df['infer_distance'] = res_df.apply(lambda x: mydict[x['loadingOrder']] if x['loadingOrder'] in mydict else x['linear_distance'], axis=1)
    # 表示推断出的距离，如果在上面的字典中的话，那么就是存在字典中的距离，否则就是直线距离代替。
    # 这三个一样的
    result=res_df[['loadingOrder','infer_distance']]
    result.to_csv('infer_distance.csv')
    '''
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
        'booster': 'gbtree',
        'objective': 'reg:gamma',
        'gamma': 0.1,
        'max_depth': 5,
        'lamda': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': seed,
        'nthread': 8,
        'eval_meric': 'rmse'
    }
    # train
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train[pred].iloc[train_idx], train[label].iloc[train_idx]
        valid_x, valid_y = train[pred].iloc[valid_idx], train[label].iloc[valid_idx]
        # 数据加载
        #n_train = xgb.DMatrix(train_x, label=train_y)
        #n_valid = xgb.DMatrix(valid_x, label=valid_y)

        #xgbModel = XGBRegressor(
        #    max_depth=30,
        #    learning_rate=0.1,
        #    n_estimators=5,
        #    objective='reg:squarederror',
        #    booster='gbtree',
        #    gamma=0.1,
        #    seed=seed
        #)
        #xgbModel.fit(train_x, train_y, verbose=True)
        #train_pred[valid_idx] = xgbModel.predict(valid_x)
        #test_pred += xgbModel.predict(test[pred]) / fold.n_splits
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
    #NROWS = 20000000
    train_data = pd.read_csv(train_gps_path)
    test_data = pd.read_csv(test_data_path)
    print(train_data.columns)
    print(train_data.shape)
    train_data = get_data(train_data, mode='train')
    test_data = get_data(test_data, mode='test')
    #temp_train_data=train_data.copy()
    train = get_feature_train(train_data)
    #temp_train=train.copy()
    calculate_difference(train_data,train)
    test = get_feature_test(test_data, port_data_path,train,train_data)
    print(train.columns)
    print(test.columns)
    print(train.shape)
    features = [c for c in train.columns if c not in ['count', 'label', 'loadingOrder','anchor_7_15','anchor_ratio_7_15','speed_mse','linear_distance']]

    #train.to_csv('train.csv')
    #test.to_csv('test.csv')
    print('FEATURES:'+str(features))
    '''
    train['anchor_0_6'] = train['anchor_0_6'].astype(float)
    train['anchor_7_15'] = train['anchor_7_15'].astype(float)
    train['anchor_ratio_0_6'] = train['anchor_ratio_0_6'].astype(float)
    train['anchor_ratio_7_15'] = train['anchor_ratio_7_15'].astype(float)
    train['distance'] = train['distance'].astype(float)
    train['mean_speed'] = train['mean_speed'].astype(float)
    train['speed_mse'] = train['speed_mse'].astype(float)
    train['mean_speed_skip0'] = train['mean_speed_skip0'].astype(float)
    train['speed_mse_skip0'] = train['speed_mse_skip0'].astype(float)
    train['first_longitude'] = train['first_longitude'].astype(float)
    train['first_latitude'] = train['first_latitude'].astype(float)
    train['last_longitude'] = train['last_longitude'].astype(float)
    train['last_latitude'] = train['last_latitude'].astype(float)

    test['anchor_0_6'] = test['anchor_0_6'].astype(float)
    test['anchor_7_15'] = test['anchor_7_15'].astype(float)
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
    '''

    result = build_model(train, test, features, 'label', is_shuffle=True)
    result.to_csv('result-061903.csv')
    # 构建并训练模型，result就是预测出的消耗的时间，再加上起始时间就是ETA；
    test_data = test_data.merge(result, on='loadingOrder', how='left')
    test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x: pd.Timedelta(seconds=x))).apply(
        lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    test_data['timestamp'] = test_data['temp_timestamp']
    # 整理columns顺序
    result = test_data[
        ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
         'creatDate']]
    result.to_csv('testout-061903.csv', index=False)


if __name__ == '__main__':
    main()