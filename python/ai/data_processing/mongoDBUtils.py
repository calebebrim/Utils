from pymongo import MongoClient, ASCENDING

import datetime
import re


def parseFieldToDate(database, collection_name, field, format="%m/%d/%Y", batch=100,selector=None):
    
    if not selector: 
        selector = {}
        selector[field] = {'$regex': re.sub(r'\%[a-zA-Z]', '\d+?', format)}

    parseField(database, collection_name, field, lambda x: datetime.datetime.strptime(x, format),
               batch=batch, selector=selector)
    
def parseFieldToInt(database,collection_name,field,batch,selector):
    parseField(database=database,collection_name=collection_name,
                field=field,selector=selector,parser= lambda x: int(float(x)))

def parseFieldToString(database, collection_name, field, batch, selector):
    parseField(database=database, collection_name=collection_name,
               field=field, selector=selector, parser=lambda x: str(x))

def parseFieldAsExcelDate(database, collection_name, field,selector=None, batch=100):
    if selector is None:
        selector = {field: re.compile('^\d+$')}
    parseField(database, collection_name, field, lambda x:
        datetime.datetime.strptime('12/31/1899', "%m/%d/%Y") 
               + datetime.timedelta(days=int(x)), selector=selector, batch=batch)

def parseField(database, collection_name, field, parser, selector={'$type': 'string', '$ne': ''}, batch=100):
    client = MongoClient('localhost', 27017)
    collection = client[database][collection_name]
    
    print('Preparing to process: ')
    print('Batch: ',batch)
    print('Collection: ',collection_name)
    print('Database: ',database)
    print('Parser: ',parser)
    print('Field: ',field)
    try:
        filter = selector

        # filter[field] = selector
        print('Selector: ', filter)

        # collection.create_index([(collection_name, ASCENDING)], unique=True)
        still_have_data = True
        counter = 0
        cursor = collection.find(filter)
        # print('Cursor Count: ', cursor.count())

        while True:
            # print('.')
            try:
                doc = cursor.next()
                if counter%batch == 0:
                    print('Processing: ',counter)
                counter+=1
                still_have_data = True
                try:

                    parsed = parser(doc[field])
                    updt = {'$set': {field: parsed}}
                    if type(doc['_id']) is dict:
                        select = {}
                        for key in doc['_id'].keys():
                            select['_id.{}'.format(key)] = doc['_id'][key]
                    else:    
                        select = {'_id': doc['_id']}
                    # print(updt)
                    # print(select)
                    collection.update(select, updt)
                except Exception as ex:
                    print(ex)
                    print(counter,' - Going to the next row...')
                
            except Exception as exc:
                print(exc)
                if not still_have_data: 
                    print('Nothing to process.')
                    break
                else:
                    print('Going to the next query...')
                    still_have_data = False
                    cursor = collection.find(filter)
                    # print('Cursor Count: ', cursor.count())

                        
    except Exception as ex:
        print(ex)
        
    print('done!')

def parseField_problem(database, collection_name, field, parser, selector={'$type': 'string', '$ne': ''}, batch=100):
    client = MongoClient('localhost', 27017)
    collection = client[database][collection_name]

    print('Preparing to process: ')
    print('Batch: ', batch)
    print('Collection: ', collection_name)
    print('Database: ', database)
    print('Parser: ', parser)
    try:
        
        project = {}
        project[field] = 1
        
        filter = selector
        
        print('Selector: ', filter)
        print('Fields: ', project)
        
        
        i = 0
        while(True):
            print('Processing row ', i, ' to ', i+batch)
            for doc in collection.find(selector,project)[0:batch]:
                # print('. ')
                try:
                    parsed = parser(doc[field])
                    updt = {'$set': {field: parsed}}
                    select = {'_id': doc['_id']}
                    collection.update(select, updt)
                except Exception as ex:
                    print('Parser Error: ',ex)
                    print(doc)
            # i+=batch

    except Exception as ex:
        raise ex 

    print('done!')

def calculate(database, collection_name, calculator,result='result', selector={}, batch=100):
    client = MongoClient('localhost', 27017)
    collection = client[database][collection_name]

    print('Preparing to process: ')
    print('Batch: ', batch)
    print('Collection: ', collection_name)
    print('Database: ', database)
    
    
    try:
        filter = selector

        # filter[field] = selector
        print('Selector: ', filter)

        # collection.create_index([(collection_name, ASCENDING)], unique=True)
        still_have_data = True
        counter = 0
        cursor = collection.find(filter)
        # print('Cursor Count: ', cursor.count())

        while True:
            # print('.')
            try:
                doc = cursor.next()
                if counter % batch == 0:
                    print('Processing: ', counter)
                counter += 1
                still_have_data = True
                try:

                    parsed = calculator(doc)
                    updt = {'$set': {result: parsed}}
                    if type(doc['_id']) is dict:
                        select = {}
                        for key in doc['_id'].keys():
                            select['_id.{}'.format(key)] = doc['_id'][key]
                    else:
                        select = {'_id': doc['_id']}
                    # print(updt)
                    # print(select)
                    collection.update(select, updt)
                except Exception as ex:
                    print(ex)
                    print(counter, ' - Going to the next row...')

            except Exception as exc:
                print(exc)
                if not still_have_data:
                    print('Nothing to process.')
                    break
                else:
                    print('Going to the next query...')
                    still_have_data = False
                    cursor = collection.find(filter)
                    # print('Cursor Count: ', cursor.count())

    except Exception as ex:
        print(ex)

    print('done!')

def hashAll(database, collection_name,field=None,batch=100):
    client = MongoClient('localhost', 27017)
    collection = client[database][collection_name]

    print('Preparing to process: ')
    print('Batch: ', batch)
    print('Collection: ', collection_name)
    print('Database: ', database)

    mapping = {}
    i = 0
    try:
        cursor = collection.find()
        while(True):
            if i%batch == 0:
                print('Processing row ', i,)
            doc = cursor.next()
            mapper(doc,mapping)
            i+=1
    except Exception as ex:
        print(ex)
    print('done!')
    return mapping, i
    
def mapper(doc,mapping,prefix=None,id=None):
    for key in doc.keys():
        mapkey =  prefix+'.'+key if prefix is not None else key
           
        if type(doc[key]) is dict:
            mapper(doc[key], mapping, prefix=mapkey, id=doc['_id'])
        elif type(doc[key]) is list:
            for idx, val in enumerate(doc[key]):
                mapper(val, mapping, prefix=mapkey+'.'+str(idx),id=doc['_id'])
        else:
            
            if(mapkey not in mapping.keys()):
                mapping[mapkey] = {}
            if(doc[key] not in mapping[mapkey].keys()):
                mapping[mapkey][doc[key]] = []
            try:
                if id is None:
                    mapping[mapkey][doc[key]].append(doc['_id'])                
                else:
                    mapping[mapkey][doc[key]].append(id)
            except Exception as ex:
                mapping[mapkey][doc[key]].append(doc)

def docToSelect(doc,prefix=None,select=None):
    if type(doc) is list:
        result = []
        for d in doc:
            result.append(docToSelect(d))
        return result
    if select is None:
        select = {}
    for key in doc.keys():
        mapkey = prefix+'.'+key if prefix is not None else key
        if type(doc[key]) is dict:
         return docToSelect(doc[key],prefix=mapkey,select=select)
        elif type(doc[key]) is list:
            select[mapkey] = []
            for idx, val in enumerate(doc[key]):
                select[mapkey].append(docToSelect(val, prefix=mapkey+'.'+str(idx)))
            
        else:   
            select[mapkey] = doc[key]
    
    return select
    
def selectByIdArray(database,collection_name,map):
    client = MongoClient('localhost', 27017)
    collection = client[database][collection_name]
    results={}
    mln = len(map)
    count = 0
    for date in map:
        if (count*100/mln)%10==0:
            print((count*100/mln),"% done...")
        selects = map[date]
        if date not in results.keys():
            results[date] = []
        for select in selects:

            results[date].append(list(collection.find(docToSelect({'_id':select})))[0])
        count+=1
    print('completed!')
    return results

