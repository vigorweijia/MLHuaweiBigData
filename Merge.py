import pandas as pd

def merge():
    data = pd.read_csv('sshWash/chunk_0.csv')
    #data2 = pd.read_csv('wxyWash/chunck_1.csv',nrows=10)
    #merge_test = pd.concat([data1,data2],ignore_index=True)
    #del merge_test['Unnamed: 0']
    #merge_test.to_csv('merge/merge_test.csv')
    for i in range(1, 30):
        chunkPath = 'sshWash/chunk_'+str(i)+'.csv'
        chunk = pd.read_csv(chunkPath)
        data = pd.concat([data,chunk],ignore_index=True)
        print("merging "+chunkPath+"...")
    data.to_csv('merge/merged_train_data.csv',index=False)


if __name__=='__main__':
    merge()