import pandas as pd
from tqdm import tqdm
import numpy as np

train_gps_path = 'event_port/train0523.csv'
test_data_path = 'event_port/A_testData0531.csv'
order_data_path = 'event_port/loadingOrderEvent.csv'
port_data_path = 'event_port/port.csv'


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


def WashTraceMoreThan2():
    for i in range(0, 30):
        filePath = 'washed/chunck_'+str(i)+'.csv'
        data = pd.read_csv(filePath)
        data = data.astype(str)
        #data = data[data['TRANSPORT_TRACE'].str.count('-') == 1]
        data = data[~data['direction'].str.contains('-1')]
        data = data[~data['vesselNextport'].str.contains('nan')]
        outputFileName = 'washed2/chunck_'+str(i)+'.csv'
        print(outputFileName)
        data.to_csv(outputFileName)


if __name__=='__main__':
    WashTraceMoreThan2()