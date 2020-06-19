import pandas as pd
import datetime

def main():
    data = pd.read_csv('testout-061900.csv')
    data['ETA'] = pd.to_datetime(data['ETA'], infer_datetime_format=True)
    data['ETA'] = data['ETA'] + datetime.timedelta(days=7, hours=4)
    data['ETA'] = data['ETA'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
    data.to_csv('AddSevenDay.csv', index=False)

if __name__=='__main__':
    main()