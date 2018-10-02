
def addToMongo(dataframe, collection,database):
    from pymongo import MongoClient

    client = MongoClient('localhost', 27017)
    collection = client[database][collection]
    import json
    records = json.loads(dataframe.T.to_json()).values()
    print(records)
    try:
        collection.insert(records)
    except Exception as ex:
        print(ex)
        # print(records)
    pass


def removeDotsFromName(dataframe):
    r = {}
    for key in dataframe.keys():
        if '.' in key:
            r[key] = key.replace('.', ' ')
    return dataframe.rename(index=str, columns=r)


def hasWeirdChars(dataframe, chars=''):
    import numpy as np
    data = dataframe
    cols = []
    rx = '['+chars+']'
    row_select = np.zeros(len(data),dtype = bool)
    for d in data:
        if (data[d].dtype == 'object'):
            hwc = data[d].str.contains(rx)
            if (hwc.any()):
                cols.append(d)
                row_select = row_select | hwc
    return cols,row_select


def describeColumns(data,unique=False):
   for i in data.keys():
        if unique: 
            print(data[i].unique())
            print('Unique: ', data[i].unique().size)
        print('HasNaN: ', data[i].isnull().any())
        # if(data[i].dtype == 'object'):
        #     print('IsNumeric: ', data[i].isnumeric())
        print(data[i].describe([.25, .5, .75]))
        print('-------------------------------------')
        print(' ')

def hashColumns(data,columns=None):
    import hashlib

    def parser(x): return hashlib.md5(str(x).encode()).hexdigest()
    data = data.copy()
    for i in data.keys():
        if (i in included_column):
            data[i] = data[i].apply(parser)
    return data