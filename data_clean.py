import numpy as np
import pandas as pd

port_hash_path = 'event_port/port_archived.csv'


def Wash():
    #CHUNKSIZE = 100000
    #NROWS = CHUNKSIZE * 10
    #data = pd.read_csv(train_gps_path, header=None)
    port_map = pd.read_csv(port_hash_path)
    # port_hash_map.columns = ['num', 'trans_name', 'longtitude', 'latitude',
    #                        'country', 'state', 'real_name', 'avg_lon', 'avg_lat']
    port_hash_map = {}
    for index, row in port_map.iterrows():
        port_hash_map[row['TRANS_NODE_NAME']] = row['REAL_PORT_NAME']
    test_port = np.loadtxt('event_port/test_port.csv', dtype=str)
    for i in range(len(test_port)):
        test_port[i] = port_hash_map[test_port[i]]

    for i in range(20,30):
        chunk_path = 'washed2/chunck_'+str(i)+'.csv'
        chunk = pd.read_csv(chunk_path)
        del chunk['Unnamed: 0']
        chunk.columns = ['a','loadingOrder', 'carrierName', 'timestamp', 'longitude',
                         'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
                         'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
        del chunk['a']
        chunk = chunk.astype(str)
        chunk = chunk[chunk['TRANSPORT_TRACE'].str.contains('-')]
        # 以下是清洗路由中出现的所有港口都和test集里没有关系的样本
        chunk['mark'] = chunk.apply(lambda x: 1 if Wash_trace_occur_in_test(x, port_hash_map, test_port) else 0, axis=1)
        chunk = chunk[chunk['mark'] == 1]
        chunk = chunk.drop('mark', axis=1)

        # 以下是清洗路由长度大于2的训练样本
        chunk['mark'] = chunk.apply(lambda x: 1 if x['TRANSPORT_TRACE'].find('-') == x['TRANSPORT_TRACE'].rfind('-') else 0, axis=1)
        chunk = chunk[chunk['mark'] == 1]
        chunk = chunk.drop('mark', axis=1)

        filePath = 'wxyWash/chunk_' + str(i) + '.csv'
        print('generate ' + filePath + '...')
        chunk.to_csv(filePath, index=False)


def Wash_trace_occur_in_test(row, port_hash_map, test_port):
    trace = row['TRANSPORT_TRACE']
    flag = False
    start = 0
    idx = trace.find('-')
    while idx != -1:
        if trace[start:idx] in port_hash_map and port_hash_map[trace[start:idx]] in test_port:
            flag = True
        start = idx + 1
        idx = trace.find('-', start)
    if trace[start:] in port_hash_map and port_hash_map[trace[start:]] in test_port:
        flag = True
    return flag


def main():
    Wash()


if __name__ == '__main__':
    main()
