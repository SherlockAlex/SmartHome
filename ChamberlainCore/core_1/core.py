import paho.mqtt.client as mqtt
import threading
import datatool as dt
import json
import pandas as pd
import brain
import numpy as np
from datatool import DeviceCoder
from datatool import DataCoder
import datetime
from weather import Weather
import subprocess

#   智能管家处理器
#   接受App发送过来的Chamberlain主题消息
#   发送请求消息给智能管家app

class Core():
    def __init__(self):
        #self.serverURL = "47.236.121.153"

        self.serverURL = "127.0.0.1"
        self.port = 1883
        self.clientID = "Chamberlain_Core"
        self.username = "Chamberlain_Core"
        self.password = "Chamberlain_Core"
        
        self.publish_list = []

        self.csv_data = dt.CSVSequnceData("./data.csv",["device_name","variable","room","year","month","day","hour","minute","home","data_name","data_type","data_value"],step = 1)
        self.csv_data.set_preprocess_callback(self.prepare)
        self.csv_data.set_encode_callback(self.encode)

        # 创建设备编码器
        self.device_coder = DeviceCoder()

        # 创建变量编码器
        self.data_coder = DataCoder()

        # 创建天气
        self.weather = Weather()

        
        pass

    def run(self):

        self.client = mqtt.Client(client_id = self.clientID)
        self.client.username_pw_set(username = self.username,password = self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(self.serverURL,self.port,60)

        self.sub_topic = "Chamberlain"
        self.client.subscribe(self.sub_topic)
        self.pub_topic_to_device = "Core"
        self.pub_topic_to_user = "Core_Data"

        # 创建大脑
        self.create_brain()

        check = input("加载?(y:加载|n:训练)")
        if check == "y":
            self.net.load("javis.h5")
        elif check == "n":
            self.train_brain()
        self.test_brain()

        send_thread = threading.Thread(target=self.send_thread)
        send_thread.start()

        self.client.loop_forever()
        pass

    def on_connect(self,client,userdata,flags,rc):
        #print("Connected with result code "+ str(rc))
        pass

    def on_message(self,client,userdata,msg):
        payload = msg.payload.decode()
        json_data = json.loads(payload)

        print("Received message: ",json_data)
        
        message = {}
        message["device_name"] = json_data["device_name"]
        message["room"] = json_data["room"]
        message["data_name"] = json_data["data_name"]
        message["data_type"] = json_data["data_type"]
        message["data_value"] = json_data["data_value"]
        message_json = json.dumps(message)
        self.client.publish(self.pub_topic_to_device,message_json)

        # 将数据写入到data.csv文件中
        dt.write_json_to_file("test.csv",json_data=json_data)

        # 同时通过神经网络进行预测用户可能的下一个操作，并发送

        df = dt.json_to_data_frame(json_data)
        input = self.csv_data.create_sequence(df=df)
        input = input.reshape(input.shape[0],input.shape[2])

        outputs = self.net.compute(input=input)
        sequences = self.net.decode(outputs=outputs)

        for sequence in sequences:
            data = {
                "device_name":sequence[0][0]
                ,"room":sequence[0][1]
                ,"data_name":sequence[2][0]
                ,"data_type":sequence[2][1]
                ,"data_value":float(sequence[3])    # float32数据，但是json无法直接序列化
            }

            public_json = json.dumps(data)

            pub_msg = (sequence[1],public_json)
            self.publish_list.append(pub_msg)
            print("predict:",sequence)
            #设置定时发送函数
            #self.client.publish(self.pub_topic,sequence)
        
        
        pass

    def send_thread(self):
        
        while True:
            #发布消息
            #client.publish(pub_topic,str(msg))

            # 发送天气消息给用户
            #weather_json = self.weather.get_now()
            #self.client.publish(self.pub_topic_to_user,str(weather_json))
            
            #在这里面计数，到了一定数量就训练模型
            now = datetime.datetime.now()
            for process in self.publish_list:
                pub_time = process[0]
                msg = process[1]
                if now.hour == pub_time[0] and now.minute == pub_time[1]:
                    self.client.publish(self.pub_topic_to_device,msg)
                    print(msg,"定时发送")
                    self.publish_list.remove(process)

            pass
        pass

    pass

    def prepare(self,df):
        
        df["device_x1"],df["device_x2"] = zip(*df.apply(lambda row:self.device_coder.fit_transform(id = row["device_name"],controlled=row["variable"],room=row["room"]),axis = 1))

        df["date_vec_x"],df["date_vec_y"] = zip(*df.apply(lambda row:dt.date_to_vector(year=row["year"],month=row["month"],day=row["day"]),axis = 1))
        df["time_vec_x"],df["time_vec_y"] = zip(*df.apply(lambda row:dt.time_to_vector(hour=row["hour"],minute=row["minute"]),axis = 1))

        df["home"] = df["home"].map({"yes":1,"no":-1})
        
        df["data_x1"],df["data_x2"] = zip(*df.apply(lambda row:self.data_coder.fit_transform(data_name=row["data_name"],data_type=row["data_type"]),axis = 1))
        # 对于输出项数据，我们需要将其传为向量
        df["value_x1"],df["value_x2"] = zip(*df.apply(lambda row:self.data_coder.transform(row["data_type"],row["data_value"]),axis = 1))
        df = df.drop(["device_name","variable","room","year","month","day","home","hour","minute","data_name","data_type","data_value"],axis=1)
        
        print(df.head())
        return df
    pass

    def encode(self,dataset):
        return dataset
    
    def create_brain(self):
        self.csv_data.load_file()
        x_data,y_data = self.csv_data.create_train_sequence()
        x_data = x_data.reshape((x_data.shape[0],x_data.shape[2]))
        x_train,y_train,x_test,y_test = dt.slice_train_test_data(x_data=x_data,y_data=y_data,slice_rate=0.8)

        #input_dim = (x_train.shape[1],x_train.shape[2])
        input_dim = (x_train.shape[1],)
        self.net = brain.Brain(
            device_coder=self.device_coder
            ,data_coder=self.data_coder
            ,input_dim=input_dim
            ,output_dim=y_train.shape[1]
        )
        
        pass

    def train_brain(self):
        self.csv_data.load_file()
        x_data,y_data = self.csv_data.create_train_sequence()
        x_data = x_data.reshape((x_data.shape[0],x_data.shape[2]))
        x_train,y_train,x_test,y_test = dt.slice_train_test_data(x_data=x_data,y_data=y_data,slice_rate=0.8)
        self.net.train(x_train,y_train,50,32,(x_test,y_test))
        self.net.save("javis.h5")
        pass

    def test_brain(self):
        #处理数据并获得df
        test_info1 = ["Name","shine","living_room",2024,2,25,17,55,"yes","switch","bool",1]
        test_info2 = ["Name","shine","living_room",2024,6,6,17,55,"yes","switch","bool",1]
        df = self.csv_data.create_dataframe([test_info1,test_info2])

        #将获得的数据加工出来，给神经网络预测
        input = self.csv_data.create_sequence(df=df)
        input = input.reshape(input.shape[0],input.shape[2])
        output = self.net.compute(input=input)

        #output = output.reshape(output.shape[0],output.shape[2])

        print(output)
        sequences = self.net.decode(output)
        for sequence in sequences:
            print(sequence)
        pass

core = Core()
core.run()
