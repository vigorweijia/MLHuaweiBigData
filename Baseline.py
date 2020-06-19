import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import KFold
import lightgbm as lgb
#lightgbm简单理解就是一种很好用的决策树集成算法；
import warnings
warnings.filterwarnings('ignore')
# baseline只用到gps定位数据，即train_gps_path
train_gps_path = 'sshWash/washed_train_data.csv'
test_data_path = 'event_port/A_testData0531.csv'
order_data_path = 'event_port/loadingOrderEvent.csv'
port_data_path = 'event_port/port.csv'
# 取前1000000行
debug = False
NDATA = 1000000


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


# 代码参考：https://github.com/juzstu/TianChi_HaiYang
def get_feature(df, mode='train'):
    #df即data
    assert mode == 'train' or mode == 'test'

    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)
    #这里是先按照订单号进行排序，然后按照时间进行排序，inplace表示需要改变原来的顺序
    # 特征只选择经纬度、速度\方向
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    #按照loadingOrder进行分组后，对于经度，将同组内的相邻的两个相减，得出差值。
    #由于之前已经按照时间排过序了，因此这里就是计算两次测量之间的纬度差值
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)
    df['speed_diff'] = df.groupby('loadingOrder')['speed'].diff(1)
    df['diff_minutes'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds() // 60
    df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] <= 0.03 and x['lon_diff'] <= 0.03
                                           and x['speed_diff'] <= 0.3 and x['diff_minutes'] <= 10 else 0, axis=1)
    #axis=1，把每一行作为x
    #anchor代表抛锚次数，也就是将前后两次比较，假如经纬度的差值很小，速度差值也很小，就看做是不动了
    #这里的判断应该是有问题的，暂且忽略不计
    #判断不在行驶状态下的记录
    if mode == 'train':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(mmax='max', count='count', mmin='min').reset_index()
        #将数据按照订单号进行分组，然后求出每一组的timestamp的最大值、最小值以及count
        # 读取数据的最大值-最小值，即确认时间间隔为label
        #也就是说，起始时间和终止时间就是在港上航行的时间
        group_df['label'] = (group_df['mmax'] - group_df['mmin']).dt.total_seconds()
    elif mode == 'test':
        group_df = df.groupby('loadingOrder')['timestamp'].agg(count='count').reset_index()

    anchor_df = df.groupby('loadingOrder')['anchor'].agg('sum').reset_index()
    #按照订单号进行分组，每一组的anchor求和，即抛锚次数求和
    anchor_df.columns = ['loadingOrder', 'anchor_cnt']
    group_df = group_df.merge(anchor_df, on='loadingOrder', how='left')
    #将group_df和anchor_df合并
    group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['count']
    #求出anchor_ratio，抛锚次数/检测次数
    agg_function = ['min', 'max', 'mean', 'median']
    agg_col = ['latitude', 'longitude', 'speed', 'direction']

    group = df.groupby('loadingOrder')[agg_col].agg(agg_function).reset_index()
    #group中存储的是agg_function和agg_col的组合，一共4*4+1=17个列
    group.columns = ['loadingOrder'] + ['{}_{}'.format(i, j) for i in agg_col for j in agg_function]
    #为每一列重新命名
    group_df = group_df.merge(group, on='loadingOrder', how='left')
    #合并group和group_df
    return group_df


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
    if debug:
        train_data = pd.read_csv(train_gps_path, nrows=NDATA, header=None)
    else:
        train_data = pd.read_csv(train_gps_path, header=None)

    train_data.columns = ['a','loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']
    train_data.drop([0],inplace=True)
    #参数分别的含义：运单号、承运商名称、当前时间、经度、
    # 纬度、船舶标识符、瞬时速度、方向、船舶下一个港口、
    # 下一个港口到达时间、当前船舶航行状态、传播数据来源、船的路由
    test_data = pd.read_csv(test_data_path)


    train_data = get_data(train_data, mode='train')
    test_data = get_data(test_data, mode='test')


    train = get_feature(train_data, mode='train')
    test = get_feature(test_data, mode='test')
    features = [c for c in train.columns if c not in ['loadingOrder', 'label', 'mmin', 'mmax', 'count']]
    #feature包括比如latitude_min，等16列and anchor_cnt,anchor_ratio

    #train.to_csv('train_baseline.csv')
    #test.to_csv('test_baseline.csv')

    result = build_model(train, test, features, 'label', is_shuffle=True)
    #result.to_csv('result_baseline.csv')
    #构建并训练模型，result就是预测出的消耗的时间，再加上起始时间就是ETA；
    test_data = test_data.merge(result, on='loadingOrder', how='left')
    test_data['ETA'] = (test_data['onboardDate'] + test_data['label'].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data.drop(['direction','TRANSPORT_TRACE'],axis=1,inplace=True)
    test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
    test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
    test_data['timestamp'] = test_data['temp_timestamp']
    # 整理columns顺序
    result = test_data[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]
    result.to_csv('testout_baseline.csv')

if __name__=='__main__':
    main()