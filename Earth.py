from math import sin,radians,cos,asin,sqrt

#unit: km
def dis(lon1, lat1, lon2, lat2):
    #lon1, lat1, lon2, lat2 = map(radians, [120.209656, 36.020225, 120.298233, 36.061589])
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

if __name__=='__main__':
    #print(dis(120.209656, 36.020225, 120.298233, 36.061589))
    #print(dis(120.313284, 36.065467, 120.209656, 36.020225))
    #print(dis(120.313284, 36.065467, 0.831116, -0.3866592))
    print(dis(122.12, 29.81, -104.31, 19.09))
