import pandas as pd
import Earth

trainDataFilePath = 'event_port/train0523.csv'
portDataFilePath = 'event_port/port.csv'
nData = 100
DisThreshold = 20

portTable = {}
portIndex2Name = {}


def UnionFind(x, fa):
    if x == fa[x]:
        return x
    else:
        son = x
        while x != fa[x]:
            x = fa[x]
        while son != x:
            tmp = fa[son]
            fa[son] = x
            son = tmp
        return x


def UnionJoin(portName1, portName2, fa):
    x = portTable[portName1]
    y = portTable[portName2]
    x = UnionFind(x, fa)
    y = UnionFind(y, fa)
    if x != y:
        fa[x] = y


def IsSameLocation(lon1, lat1, lon2, lat2):
    if Earth.dis(lon1, lat1, lon2, lat2) < DisThreshold:
        return True
    else:
        return False


def ArchivePort(portList):
    portList.columns = ['TRANS_NODE_NAME', 'LONGITUDE', 'LATITUDE', 'COUNTRY', 'STATE',
                        'CITY', 'REGION', 'ADDRESS', 'PORTCODE', 'TRANSPORT_NODE_ID']
    portList.drop(portList.index[0], inplace=True)
    portList['TRANS_NODE_NAME'] = portList['TRANS_NODE_NAME'].astype(str)
    for i, name in enumerate(portList['TRANS_NODE_NAME']):
        portTable[name] = i
        portIndex2Name[i] = name
    fa = list(range(len(portList)))

    Len = len(portList)
    for i in range(1, Len):
        portNameA = portList.at[i, 'TRANS_NODE_NAME']
        portLonA = float(portList.at[i, 'LONGITUDE'])
        portLatA = float(portList.at[i, 'LATITUDE'])
        countryA = portList.at[i, 'COUNTRY']
        stateA = portList.at[i, 'STATE']
        for j in range(i, Len):
            if i == j:
                continue
            portNameB = portList.at[j, 'TRANS_NODE_NAME']
            portLonB = float(portList.at[j, 'LONGITUDE'])
            portLatB = float(portList.at[j, 'LATITUDE'])
            countryB = portList.at[j, 'COUNTRY']
            stateB = portList.at[j, 'STATE']
            if IsSameLocation(portLonA, portLatA, portLonB, portLatB) and countryA == countryB and stateA == stateB:
                #print(portNameA+': '+str(portLonA)+' '+str(portLatA)+' '+portNameB+': '+str(portLonB)+' '+str(portLatB))
                UnionJoin(portNameA, portNameB, fa)

    portList['REAL_PORT_NAME'] = None
    portList['AVG_LON'] = None
    portList['AVG_LAT'] = None

    for i in range(1, Len):
        sumLon = float(portList.at[i, 'LONGITUDE'])
        sumLat = float(portList.at[i, 'LATITUDE'])
        x = portTable[portList.at[i, 'TRANS_NODE_NAME']]
        cnt = 1
        for j in range(1, Len):
            y = portTable[portList.at[j, 'TRANS_NODE_NAME']]
            if UnionFind(x, fa) == UnionFind(y, fa):
                sumLon += float(portList.at[j, 'LONGITUDE'])
                sumLat += float(portList.at[j, 'LATITUDE'])
                cnt += 1
        avgLon = float(sumLon/cnt)
        avgLat = float(sumLat/cnt)
        portList.at[i, 'AVG_LON'] = avgLon
        portList.at[i, 'AVG_LAT'] = avgLat

    for i in range(1, Len):
        portName = portList.at[i, 'TRANS_NODE_NAME']
        realPortName = UnionFind(portTable[portName], fa)
        portList.at[i, 'REAL_PORT_NAME'] = portIndex2Name[realPortName]

    #del portList['COUNTRY']
    #del portList['STATE']
    portList['REAL_PORT_NAME'] = portList['REAL_PORT_NAME'].astype(str)
    del portList['CITY']
    del portList['REGION']
    del portList['ADDRESS']
    del portList['PORTCODE']
    del portList['TRANSPORT_NODE_ID']
    portList.to_csv('port_archived.csv')


if __name__=='__main__':
    #train_data = pd.read_csv(trainDataFilePath, nrows=nData, header=None)
    #train_data.columns = ['loadingOrder', 'carrierName', 'timestamp',
    #                      'longitude', 'latitude', 'vesselMMSL',
    #                      'speed', 'direction', 'vesselNextport',
    #                      'vesselNextportETA', 'vesselStatus', 'vesselDataource',
    #                      'TRANSPORT_TRACE']
    #train_data.to_csv('frontData.csv')
    portData = pd.read_csv(portDataFilePath, header=None)

    #print(portData)
    ArchivePort(portData)
