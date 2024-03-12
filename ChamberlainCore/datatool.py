import os
import pandas as pd
import json
from sklearn import preprocessing
import numpy as np
from datetime import datetime
import math
import hashlib

#数据的处理工具

#将时间映射为一个2维矢量
#如果时间输入的特征变化过少
#会出现特征空间的不连续

def date_to_vector(year,month,day):
    day_count = {
        0:31,
        1:28,
        2:31,
        3:30,
        4:31,
        5:30,
        6:31,
        7:31,
        8:30,
        9:31,
        10:30,
        11:31
    }

    if year%400 == 0:
        day_count[1] = 29
        pass

    #计算今年是共有几天
    total_day = 0
    for i in range(12):
        total_day = total_day + day_count[i]
    
    current_day = 0
    #计算当前天是一年之中的第几天
    for i in range(month):
        current_day = current_day+day_count[i]
    current_day = current_day + day

    radian = (2*math.pi*current_day)/total_day

    return np.array([math.cos(radian),math.sin(radian)])

    pass

def time_to_vector(hour,minute):
    _total_minute = 24*60
    _current_minute = hour*60 + minute
    radian = (2*math.pi*_current_minute)/_total_minute
    return np.array([math.cos(radian),math.sin(radian)])
    pass

def vector_to_time(vector):
    # 定义时间编码的范围，这里假设时间范围是0点到23点59分
    hours = range(24)
    minutes = range(60)
    
    # 将神经网络输出的向量标准化
    vector = vector / np.linalg.norm(vector)
    
    # 初始化最高相似度和对应时间
    max_similarity = -1
    predicted_time = None
    
    # 遍历所有时间编码向量，计算余弦相似度
    for hour in hours:
        for minute in minutes:
            time_vector = time_to_vector(hour, minute)
            time_vector = time_vector / np.linalg.norm(time_vector)
            similarity = np.dot(vector, time_vector)
            
            # 更新最高相似度和对应时间
            if similarity > max_similarity:
                max_similarity = similarity
                predicted_time = (hour, minute)
    
    return predicted_time

def _check_file_and_json(filepath,json_data):
    file_exists = os.path.isfile(filepath)
    if not file_exists:
        return False
    
    df = pd.read_csv(filepath,nrows=1)
    csv_columns = df.columns.tolist()
    json_columns = list(json_data.keys())

    #检测json的属性列表是否相同
    columns_match = (csv_columns==json_columns)

    return columns_match

def json_to_data_frame(json_data):
    try:
        df = pd.DataFrame(json_data)
        return df
    finally:
        series = pd.Series(json_data)
        df = pd.DataFrame([series])
        return df
    pass

def write_json_to_file(filename,json_data):
    df = json_to_data_frame(json_data=json_data)
    if _check_file_and_json(filepath=filename,json_data=json_data):
        #表示文件存在追加
        df.to_csv(filename,mode="a",header=False,index=False)
        return
    df.to_csv(filename,index=False)
    pass

def slice_train_test_data(x_data,y_data,slice_rate):
    train_size = int(len(x_data)*slice_rate)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]

    x_test = x_data[train_size:]
    y_test = y_data[train_size:]

    return x_train,y_train,x_test,y_test

def slice_msk_data(x_data,y_data,slice_rate):
    index = [i for i in range(len(x_data))]
    np.random.shuffle(index)

    x_data = x_data[index]
    y_data = y_data[index]

    train_size = int(len(x_data)*slice_rate)
    x_train = x_data[:train_size]
    y_train = y_data[:train_size]

    x_test = x_data[train_size:]
    y_test = y_data[train_size:]

    msk = np.random.randn(len(x_data))<0.8
    
    x_train = x_data[msk]
    y_train = y_data[msk]

    x_test = x_data[~msk]
    y_test = y_data[~msk]

    return x_train,y_train,x_test,y_test
    pass

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return similarity

def slide_sequence(step,dataset):
        x_data,y_data = [],[]
        length = len(dataset)
        for i in range(length - step):
            x_temp = dataset[i:(i+step),:]
            y_temp = dataset[(i+step),:]
            x_data.append(x_temp)
            y_data.append(y_temp)

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data,y_data
        pass

def fixed_hash(s):
    sha256 = hashlib.sha256()
    sha256.update(s.encode('utf-8'))
    hash_bytes = sha256.digest()
    hash_int = int.from_bytes(hash_bytes,byteorder="big")
    return hash_int

class CSVData():
    def __init__(self,filename,selected_columns,feature_columns,label_columns):
        self.filename = filename
        self.selected_columns = selected_columns
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        pass

    def load_file(self):
        self.df = pd.read_csv(self.filename)
        self.df = self.df[self.selected_columns]
        
        pass
    
    # callback(df):对数据进行填充，map等,返回处理过的df
    def set_prepare_callback(self,callback):
        self._prepare_callback = callback
        pass

    def create_data(self,df):
        df = self._prepare_callback(df)
        df_features = df[self.feature_columns]
        df_labels = df[self.label_columns]
        #区分出一个函数
        x_data = df_features.values
        y_data = df_labels.values

        return x_data,y_data
        pass

    def create_dataframe(self,value):
        df = pd.DataFrame(value,columns = self.selected_columns)
        return df
        pass
    
    def create_train_data(self):
        return self.create_data(self.df)
        pass

    pass

#序列化数据
class CSVSequnceData():
    def __init__(self,filename,columns,step):
        self.filename = filename
        self.step=step      #采用步长
        self.columns=columns
        pass
    
    def load_file(self):
        self.df = pd.read_csv(self.filename)
        self.df=self.df[self.columns]
        pass

    def create_dataframe(self,value):
        df = pd.DataFrame(value,columns = self.columns)
        return df
        pass

    def set_preprocess_callback(self,callback):
        self.preprocess_callback = callback
        pass

    def set_encode_callback(self,callback):
        self.encode_callback = callback
        pass

    def _preprocess(self,df):
        df = self.preprocess_callback(df)
        #scale = preprocessing.MinMaxScaler(feature_range=features_range)
        dataset = df.values
        #dataset = scale.fit_transform(dataset)
        return dataset
        pass

    def preprocess(self):
        return self._preprocess(self.df)

    def create_train_sequence(self):
        dataset = self._preprocess(self.df)
        dataset = self.encode_callback(dataset)
        return slide_sequence(step=self.step,dataset=dataset)

    def create_sequence(self,df):
        dataset = self._preprocess(df=df)
        dataset = self.encode_callback(dataset)
        return dataset.reshape(-1,1,dataset.shape[1])
    pass

class DeviceCoder():
    def __init__(self):
        self._embedding = {}
        pass

    def fit_transform(self,id,controlled,room):
        '''
        一个简单的理论是
        设备信息作为一个智能家居的环境状态
        其本质上是对其进行向量化

        我们可以对每个单独的设备信息项进行一个2维向量化
        同时利用hash值是int数，并且cos(n),sin(n)是N to (0,1)的一一映射

        我们可以利用这两个性质将设备信息项对应到矢量空间的一个基矢量

        而我们假定设备的每个信息项都作为设备矢量的基矢量
        那么我们的设备矢量将会是基矢量的加权求和
        那么我要思考这些权值的意义

        我们希望我们的权值都是属于(0,1)之间的数，同时表示概率
        其大小会等于我们设备矢量和具体基矢量的内积，这个值越大，
        表示这个信息特征对于设备而言贡献更大，意味着这个特征对于设备更重要

        到此，我们通过数学的语言阐述了重要性的定义


        但是
        '''

        id_code = fixed_hash(id)
        controlled_code = fixed_hash(controlled)
        room_code = fixed_hash(room)

        e1 =  np.array([math.cos(id_code),math.sin(id_code)])
        e2 = np.array([math.cos(controlled_code),math.sin(controlled_code)])
        e3 = np.array([math.cos(room_code),math.sin(room_code)])

        # 根据事物属性公设，设备的环境控制变量和房间号对于设备而言更加重要
        vector = 0.069*e1 + 0.846*e2 + 0.525*e3
        self._embedding[(id,room)] = vector
        return vector
        pass

    def get_device(self, vector):
        max_sim = -1
        best_key = None

        for key, emb_vector in self._embedding.items():
            cos_sim = np.dot(vector, emb_vector) / (np.linalg.norm(vector) * np.linalg.norm(emb_vector))
            if cos_sim > max_sim:
                max_sim = cos_sim
                best_key = key

        return best_key
    
class DataCoder():
    def __init__(self):
        self._embedding = {}
        pass

    def fit_transform(self,data_name,data_type):
        name_code = fixed_hash(data_name)
        type_code = fixed_hash(data_type)

        e1 = np.array([math.cos(name_code),math.sin(name_code)])
        e2 = np.array([math.cos(type_code),math.sin(type_code)])

        vector = 0.5*e1 + 0.5*e2
        self._embedding[(data_name,data_type)] = vector
        return vector

        pass

    def transform(self,type,value):
        # 对类型数据进行编码
        
        if type == "bool":
            # (False,True)
            vector = np.array([-value,value])
            return vector
            pass
        
        radian = int(value*math.pi)
        val_vec = np.array([math.cos(radian),math.sin(radian)])
        return val_vec

        pass

    def inverse_transfrom(self,type,vector):
        if type == "bool":
            # 返回False或者True
            value = vector[0]<vector[1]
            return bool(value)
        
        cos_value = vector[0]
        sin_value = vector[1]

        radian = math.atan2(sin_value,cos_value)
        value = radian/(math.pi)

        return float(value)
        pass

    def get_data_info(self,vector):
        max_sim = -1
        best_key = None

        for key, emb_vector in self._embedding.items():
            cos_sim = np.dot(vector, emb_vector) / (np.linalg.norm(vector) * np.linalg.norm(emb_vector))
            if cos_sim > max_sim:
                max_sim = cos_sim
                best_key = key

        return best_key
        pass

    pass

class ModeCoder():
    def __init__(self):
        self._embedding={}
        self._mode_map = {
            "wake_up":1
            ,"grooming":2
            ,"morning_dining":3
            ,"away":4
            ,"homecoming":5
            ,"dining":6
            ,"movie":7
            ,"sleep":8
        }

        for item in self._mode_map.keys():
            self.one_hot_transform(item)
            pass

        pass
    
    def one_hot_transform(self,mode):
        max_dim = len(self._mode_map)
        encoded = np.full(max_dim,-1)
        encoded[self._mode_map[mode]-1] = 1
        self._embedding[mode] = encoded
        return encoded
        pass

    def fit_transform(self,mode):
        radian = fixed_hash(mode)
        x1 = math.cos(radian)
        x2 = math.sin(radian)
        index = self._mode_map[mode]
        x3 = -1
        if index%2==0:
            x3 = math.cos(index)
        else:
            x3 = math.sin(index)
        vector = np.array([x1,x2,x3])
        self._embedding[mode] = vector
        return vector
        pass

    def get_mode(self,vector):
        max_sim = -1
        best_key = None

        for key, emb_vector in self._embedding.items():
            cos_sim = np.dot(vector, emb_vector) / (np.linalg.norm(vector) * np.linalg.norm(emb_vector))
            if cos_sim > max_sim:
                max_sim = cos_sim
                best_key = key

        return best_key
        pass

    def get_similary_vector(self,vector):
        max_sim = -1
        best_target = None
        for key,emb_vector in self._embedding.items():
            cos_sim = np.dot(vector,emb_vector)/(np.linalg.norm(vector)*np.linalg.norm(emb_vector))
            if cos_sim>max_sim:
                max_sim = cos_sim
                best_target = emb_vector
        return best_target
        pass