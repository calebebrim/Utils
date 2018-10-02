import numpy as np 
import pandas as pd


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
        nll = data[i].isnull()
        print('Null:',nll.any(),' All:',nll.all()  )
        # if(data[i].dtype == 'object'):
        #     print('IsNumeric: ', data[i].isnumeric())
        print(data[i].describe())
        print('-------------------------------------')
        print(' ')

def hashColumns(data,columns=None):
    import hashlib
    def parser(x): return hashlib.md5(str(x).encode()).hexdigest()
    data = data.copy()
    for i in data.keys():
        if (i in columns):
            data[i] = data[i].apply(parser)
    return data

def autoMap(data,dtypes=[np.object,np.bool]):
    import pandas as pd
    import pickle
    reverse_map = {}
    data = data.copy()
    for k in data:
        try:
            mapping = {}
            
            if data[k].dtype in dtypes:
                print('mapping: ',k,' - ',data[k].dtype)
                uniquedata = data[k].unique()
                valueid = range(len(uniquedata))
                mapping = {}
                reverse = {}
                for vid in valueid:
                    mapping[uniquedata[vid]] = vid
                    reverse[vid] = uniquedata[vid]
                
                data[k] = data[k].replace(mapping)
                
                reverse_map[k] = reverse
                # print(mapping)
            else:
                print('Column not mapped: ',k,data[k].dtype)
        except Exception as ex:
            print(ex)
            print(mapping)
            # print(reverse_map)
            print(dtypes)
    return data,reverse_map

def fillMallping(data,reverse_map):
    import pandas as pd
    
def plotCorr(data,save_as="",plot=True,plot_title='Correlation Analysis',force_show=False,hidenan=True):
    from matplotlib import pyplot as plt

    cr = None
    if hidenan: 
        cr = data.corr().dropna(axis=0,how='all').dropna(axis=1,how='all')
    else: 
        cr = data.corr()
    plt.figure()
    ax = plt.gca()
    labels = cr.keys().values
    
    # y labels
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    
    # x labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    
    plt.imshow(cr, interpolation='nearest',cmap='Wistia')
    plt.title(plot_title)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
        # specify integer or one of preset strings, e.g.
        tick.label.set_fontsize(6) 
        tick.label.set_rotation('vertical')

    for (i, j), z in np.ndenumerate(cr.values):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize=8)
    if(save_as!=""):
        plt.savefig(save_as)
        if(force_show):
            plt.show()
    else:
        plt.show()
        plt.close()

def getGeoFromAddress(dataframe,columns=None,prefix="geo",mpc={}):
    ### columns with address or postalcode


    # pdu.getGeoFromAddress(ebm_raw,columns=['MUNICIPIO_EMPREEND','MUNICIPIOCLIENTE'...
    # import geopy as gp
    # nom = gp.Nominatim()
    
    if columns: 
        data = dataframe[columns[0]].copy()
        for i in range(1,len(columns)):
            data += ', '+dataframe[columns[i]].copy()
        

        upc = data[data.isnull()==False].str.lower().unique()
        for pc in upc:
            if(pc not in mpc):
                geocode = getGeocode(pc)
                try:    
                    mpc[pc] = geocode[0]['geometry']['location'] if pc!=None and pc!=np.nan and str(pc)!='nan' else None
                    print('{} -> {}'.format(pc,mpc[pc])) 
                except Exception as ex:
                    print(ex)
                    print(geocode)
            else: 
                print('Key Processed: {}'.format(pc))
        dataframe[prefix+'_address'] = data.map(mpc)
        dataframe[prefix+'_latitude'] = dataframe[prefix+'_address'] .apply(lambda x: x.latitude if x != None else None)
        dataframe[prefix+'_longitude'] = dataframe[prefix+'_address'] .apply(lambda x: x.longitude if x != None else None)
        return mpc

KEYINDEX = 0
def getGeocode(address, googleKey=['AIzaSyC21Br8md1UMWZuUaLrT52fCBDmaj040uQ','AIzaSyBiWW8ILY8oqKtvemU1LtOcjW6KAQtE80I','AIzaSyCfH1hV73Nu-OMxvaKdiFdz591tcXRCc8w','AIzaSyBPkCWwVfB2y7R1guyRryOeKcZ6GETQO_s'],keyindex=None,resetIndex=False,record_to=None):
    ### Google Key may expire
    global KEYINDEX
    import googlemaps
    from datetime import datetime
    if not keyindex:
        keyindex = KEYINDEX
    
    if resetIndex: 
        KEYINDEX = 0

    if type(googleKey) == str:
        googleKey = [googleKey]
    try:
        gmaps = googlemaps.Client(key=googleKey[keyindex])
        geocode_result = gmaps.geocode(address)
        return geocode_result
    except Exception as ex:
        print(ex)
        if keyindex+1<len(googleKey):
            print('Using Next Key: {}-{}'.format(str(keyindex),googleKey[keyindex]))
            KEYINDEX = keyindex+1
            getGeocode(address,googleKey=googleKey,keyindex=1+keyindex)
        else:
            print('None Keys Left')
            raise ex
    
def fromMatLabFile(path=None,varname=None):
    import numpy as np
    from scipy.io import loadmat  # this is the SciPy module that loads mat-files
    import matplotlib.pyplot as plt
    from datetime import datetime, date, time
    import pandas as pd
    
    mat = loadmat(path)  # load mat-file
    mdata = mat[varname]  # variable in mat file
    mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
    # * SciPy reads in structures as structured NumPy arrays of dtype object
    # * The size of the array is the size of the structure array, not the number
    #   elements in any particular field. The shape defaults to 2-dimensional.
    # * For convenience make a dictionary of the data using the names from dtypes
    # * Since the structure has only one element, but is 2-D, index it at [0, 0]
    ndata = {n: mdata[n][0, 0] for n in mdtype.names}
    # Reconstruct the columns of the data table from just the time series
    # Use the number of intervals to test if a field is a column or metadata
    columns = [n for n, v in ndata.iteritems() if v.size == ndata['numIntervals']]
    # now make a data frame, setting the time stamps as the index
    df = pd.DataFrame(np.concatenate([ndata[c] for c in columns], axis=1),
                    index=[datetime(*ts) for ts in ndata['timestamps']],
                    columns=columns)
    return df

def normMaxMin(values,min=0,max=1,round_int=True):
    # from sklearn.preprocessing import MinMaxScaler
    # values = np.array(values).reshape(1, -1)
    # return MinMaxScaler(feature_range=(min,max)).fit_transform(values)[0]
    try:
        X = np.array(values)
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        return X_scaled
    except Exception as ex:
        print(values,min,max)
        print(ex)

def readExcelAndSave(excel_path):        
    try:
        return pd.read_pickle(excel_path.replace('.xlsx','.pkl'))
    except Exception as ex:
        print(ex)
        print('Pickle not found reading excel file...')
        data = pd.read_excel(excel_path)
        print('Saving picke...')
        pd.to_pickle(data,excel_path.replace('.xlsx','.pkl'))
        print('Done!')
        return data


def readCsvAndSave(csv_path, encoding='UTF-8', sep=';', error_bad_lines=False,columns=None):
    pklpath = csv_path.replace('.csv', '.pkl')
    try:
        print('Looking for',pklpath)
        return pd.read_pickle(pklpath, compression='gzip')
    except Exception as ex:
        print(ex)
        
        print('Pickle not found reading CSV file...')
        data = pd.read_csv(csv_path,sep=sep,encoding=encoding,error_bad_lines=error_bad_lines,names=columns)
        print('Saving picke...')
        pd.to_pickle(data, pklpath,compression='gzip')
        print('Done!')
        return data


def saveAllIntoPKL(files_path, pickle_name, encoding='UTF-8', sep=';', error_bad_lines=False, columns=None,save_partial=True):
    data = None
    for csv_path in files_path:
        print('Processing: ',csv_path)
        if type(data) != type(None):
            
            print('Data Shape: ',data.shape)
            data2 = pd.read_csv(csv_path, sep=sep, encoding=encoding, error_bad_lines=error_bad_lines, names=columns)
            print('Data Shape: ',data2.shape)
            data = pd.concat([data,data2],ignore_index=True,verify_integrity=True)
            
        else:
            data = pd.read_csv(csv_path, sep=sep, encoding=encoding, error_bad_lines=error_bad_lines, names=columns)
        if(save_partial):
            data.to_pickle(pickle_name, compression='gzip')
            print('Data Shape: ',data.shape)
    data.to_pickle(pickle_name, compression='gzip')
    return data
