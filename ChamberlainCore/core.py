import paho.mqtt.client as mqtt
import threading
import datatool as dt
import json
import pandas as pd
import numpy as np
from weather import Weather
import time
from datetime import datetime
from scenemode import SceneMode
from device import Device

#   智能管家处理器
#   接受App发送过来的Chamberlain主题消息
#   发送请求消息给智能管家app

bool_map={
    -1:False,
    1:True
}

class Core():
    def __init__(self):
        #self.serverURL = "47.236.121.153"

        self.serverURL = "127.0.0.1"
        self.port = 1883
        self.clientID = "Chamberlain_Core"
        self.username = "Chamberlain_Core"
        self.password = "Chamberlain_Core"
        
        self.time_depended_operation = []
        
        
        # 创建天气
        self.weather = Weather()

        
        self.mode_coder = dt.ModeCoder()

        self.scene_mode = SceneMode(mode_coder=self.mode_coder)
        
        # 拥有的设备，(device_name,Device)
        self.devices = {}

        self.current_mode = "away"
        #self.device = Device(device_name="light",mode_coder=self.mode_coder)

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

        check = input("加载?(y:加载|n:训练)")
        if check == "y":
            self.scene_mode.load()
        elif check == "n":
            self.scene_mode.train()
        self.scene_mode.test()

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

        if json_data["code"] == "match_success":
            # 设备匹配成功
            self.append_device(json_data["device_name"],json_data["properties"])
            pass
        elif json_data["code"] == "property":
            message = {}
            message["device_name"] = json_data["device_name"]
            #message["room"] = json_data["room"]
            message["data_name"] = json_data["data_name"]
            message["data_type"] = json_data["data_type"]
            if message["data_type"] == "bool":
                message["data_value"] = bool_map[json_data["data_value"]]
            else:
                message["data_value"] = json_data["data_value"]
            message_json = json.dumps(message)
            self.client.publish(self.pub_topic_to_device,message_json)

            for device in self.devices.values():
                train_json = {}
                train_json["year"] = json_data["year"]
                train_json["month"] = json_data["month"]
                train_json["day"] = json_data["day"]
                train_json["mode"] = self.current_mode
                train_json["data_name"] = json_data["data_name"]
                train_json["data_type"] = json_data["data_type"]
                train_json["data_value"] = json_data["data_value"]
                device.train_self(train_json)
        
        elif json_data["code"] == "change_scene":
            # 首先更改模式
            
            if self.change_mode(year=json_data["year"],month=json_data["month"],day=json_data["day"],mode=json_data["mode"]) == False:
                # 更换模式失败，当前模式就是这个模式
                return
            # 更换模式成功
            # 将记录保存下来
            data = {}
            data["year"] = json_data["year"]
            data["month"] = json_data["month"]
            data["day"] = json_data["day"]
            data["hour"] = json_data["hour"]
            data["minute"] = json_data["minute"]
            data["mode"] = json_data["mode"]
            dt.write_json_to_file(filename="test_mode.csv",json_data=data)
            # 预测这个场景下各设备的可能状态，并操作

            # 预测下一个可能的模式(模式必须得经过训练才可以进行下面操作)
            input = self.scene_mode.json_to_input(json_data=json_data)
            output = self.scene_mode.compute(inputs=input)
            # 解析output，然后添加到待定改变列表中
            for op in output:
                self.time_depended_operation.append(op)
                print(op)
            pass

        pass

    def send_thread(self):
        
        while True:
            now = datetime.now()
            hour = now.hour
            minute = now.minute
            year = now.year
            month = now.month
            day = now.day

            for op in self.time_depended_operation:
                time_hour = op["hour"]
                time_minute = op["minute"]
                next_mode = op["mode"]

                
                if time_hour == hour and time_minute == minute:
                    if not self.change_mode(year=year,month=month,day=day,mode=next_mode):
                        # 当前预测状态没有任何改变
                        continue
                    pass
            
            time.sleep(10)
            pass
        pass

    def append_device(self,device_name,properties):
        if device_name in self.devices.keys():
            return
        device = Device(device_name=device_name,device_properties=properties,mode_coder=self.mode_coder)
        self.devices[device_name] = device
        print("添加新设备:",device_name,",属性:",properties)
        pass

    def change_mode(self,year,month,day,mode):
        print("current mode:",self.current_mode,",next mode:",mode)
        if mode == self.current_mode:
            return False
        # 执行变换模式操作
        self.current_mode = mode
        input_json = {
            "year":year
            ,"month":month
            ,"day":day
            ,"mode":mode
        }
        for device in self.devices.values():
            # 设备必须得经过训练才可以进行下面操作
            inputs = device.json_to_input(json_data = input_json)
            outputs = device.compute(inputs = inputs)
            # 发送给对应的下位机
            for output in outputs:
                print("下位机预测结果:",output)
                message = json.dumps(output)
                self.client.publish(self.pub_topic_to_device,message)

        return True
        pass

    pass

    

core = Core()
core.run()
