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
import queue
import tensorflow as tf

import pyttsx3

#   智能管家处理器
#   接受App发送过来的Chamberlain主题消息
#   发送请求消息给智能管家app

#   难题是如何预测时间

bool_map={
    -1:False,
    1:True
}

mode_text={
    "wake_up":"已经切换到起床模式"
    ,"grooming":"为你切换到洗漱模式，先生"
    ,"morning_dining":"先生，为你切换到早餐模式"
    ,"away":"为你切换到离家模式，路上小心，先生"
    ,"homecoming":"欢迎回家，先生"
    ,"dining":"为你切换到晚餐模式，先生"
    ,"movie":"为你切换到观影模式，先生"
    ,"sleep":"已经为你切换到休眠模式"
}

mode_act_map={
    "wake_up":"起床"
    ,"grooming":"洗漱"
    ,"morning_dining":"吃早餐"
    ,"away":"离家"
    ,"homecoming":"回家"
    ,"dining":"吃晚餐"
    ,"movie":"看电视"
    ,"sleep":"休息"
}

def progress_bar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█'):
    total = len(iterable)
    
    def print_progress_bar(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
        
        if iteration == total:
            print()
    
    for i, item in enumerate(iterable):
        time.sleep(0.1)
        print_progress_bar(i + 1)

class Core():
    def __init__(self):
        #self.serverURL = "47.236.121.153"

        tf.get_logger().setLevel('ERROR')

        self.serverURL = "127.0.0.1"
        self.port = 1883
        self.clientID = "Chamberlain_Core"
        self.username = "Chamberlain_Core"
        self.password = "Chamberlain_Core"
        
        self.time_depended_operation = []

        self.device_train_jsons = queue.Queue()
        # 创建天气
        self.weather = Weather()

        
        self.mode_coder = dt.ModeCoder()

        self.scene_mode = SceneMode(mode_coder=self.mode_coder)
        
        # 拥有的设备，(device_name,Device)
        self.devices = {}

        self.can_self_learn = True
        self.can_self_control = True

        self.current_enviro = {
            "mode":"away"
        }

        now = datetime.now()
        hour = now.hour
        minute = now.minute
        year = now.year
        month = now.month
        day = now.day
        self.past_op = {}
        self.past_op["year"] = year
        self.past_op["month"] = month
        self.past_op["day"] = day
        self.past_op["hour"] = hour
        self.past_op["minute"] = minute
        self.past_op["mode"] = self.current_enviro["mode"]

        self.present_op ={}
        self.present_op["year"] = year
        self.present_op["month"] = month
        self.present_op["day"] = day
        self.present_op["hour"] = hour
        self.present_op["minute"] = minute
        self.present_op["mode"] = self.current_enviro["mode"]

        #self.device = Device(device_name="light",mode_coder=self.mode_coder)

        
        
        self.predict_queue = queue.Queue()
        self.speak_queue = queue.Queue()

        self.enviro_data = {
            "temp":18
            ,"humidy":90
            ,"day_temp":18
            ,"night_temp":18
        }


        pass

    

    def run(self):

        # 初始化MQTT客户端
        self.client = mqtt.Client(client_id = self.clientID)
        self.client.username_pw_set(username = self.username,password = self.password)

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(self.serverURL,self.port,60)

        self.sub_topic = "Chamberlain"
        self.sub_enviro_topic = "device/enviro"
        self.client.subscribe(self.sub_topic)
        self.client.subscribe(self.sub_enviro_topic)
        self.pub_topic_change_device_property = "core/core/device/change_state"
        self.pub_topic_to_device = "Core"
        self.pub_topic_to_user = "core/core/data"

        try:
            self.scene_mode.load()
        except:
            pass
        
        items = list(range(100))
        progress_bar(items, prefix='加载笨笨的数据中:', suffix='加载完成', length=50)

        self.speak_queue.put("你好,我是笨笨")

        # 进入主循环
        speak_thread = threading.Thread(target=self.speak_loop)
        speak_thread.start()

        enviro_thread = threading.Thread(target=self.enviro_loop)
        enviro_thread.start()

        main_thread = threading.Thread(target=self.main_loop)
        main_thread.start()

        self.client.loop_forever()
        pass

    def on_connect(self,client,userdata,flags,rc):
        #print("Connected with result code "+ str(rc))
        pass

    def on_message(self,client,userdata,msg):

        # 首先解码收到的信息为json
        payload = msg.payload.decode()
        json_data = json.loads(payload)
        # 根据收到的json中的code判定这个消息是做什么的
        if json_data["code"] == "enviro":
            pass
        elif json_data["code"] == "can_self_learn":
            self.can_self_learn = bool(json_data["value"])
            if not self.can_self_learn:
                self.speak_queue.put("你关闭了学习生活习惯的权限")
        elif json_data["code"] == "can_self_control":
            self.can_self_control = bool(json_data["value"])
            if not self.can_self_control:
                self.speak_queue.put("先生,我没有权限帮你控制家居设备了")

        # 信息是智能管家发送过来的匹配设备成功
        elif json_data["code"] == "match_success":
            self.append_device(json_data["device_name"],json_data["properties"])
            pass

        # 信息是智能管家发送给下位机执行属性改变
        elif json_data["code"] == "property":
            message = {}
            message["device_name"] = json_data["device_name"]
            message["data_name"] = json_data["data_name"]
            message["data_type"] = json_data["data_type"]
            if message["data_type"] == "bool":
                message["data_value"] = bool_map[json_data["data_value"]]
            else:
                message["data_value"] = json_data["data_value"]
            message_json = json.dumps(message)

            # 将信息转发给下位机
            self.client.publish(self.pub_topic_to_device,message_json)
            # 对设备当前的属性值进行更新
            for device in self.devices.values():
                if device.get_device_name() == message["device_name"]:
                    device.state[message["data_name"]] = message["data_value"]
                    pass

            # 同时利用当前信息对设备神经网络进行训练
            json_data["mode"] = self.current_enviro["mode"]
            self.update_device_train_json(json_data=json_data)

            
        # 信息用于更换场景
        elif json_data["code"] == "change_scene":
            # 这边如果模型在进行自主学习的时候，会出现问题
            if self.change_mode(json_data=json_data) == False:
                return
            
            # 将用户场景切换记录保存下来
            #dt.write_json_to_file(filename="test_mode.csv",json_data=json_data)
            # 对场景切换进行学习
            # 输入上一个状态和用户当前切换的状态

            # 并不是每次都要去学
            #self.scene_mode.train_self(present_json=self.present_op,future_json=json_data,past_json=self.past_op)
            self.past_op = self.present_op
            self.present_op = json_data
            
            
            pass

        pass
    
    def predict_next_scene(self,json_data):
        if not self.can_self_control:
            return
        input = self.scene_mode.json_to_input(json_data=json_data)
        pasts,presents,futures = self.scene_mode.compute(inputs=input)
        # 解析output，然后添加到待定改变列表中
        for op in futures:
            text = "先生，计算得到你可能会在"+str(op["hour"])+"点"+str(op["minute"])+"分"+mode_act_map[op["mode"]]
            self.client.publish(self.pub_topic_to_user,text)
            self.time_depended_operation.append(op)
            
            # 寻求用户的确认，如果返回确定的话就添加到time_depended_operation
            # 否则的话返回
            self.speak_queue.put(text)
            info = {
                "hour":op["hour"]
                ,"minute":op["minute"]
                ,"mode":op["mode"]
            }
        # 解析output，然后添加到待定改变列表中
        new_input = self.scene_mode.json_to_input(json_data=futures[0])
        new_pasts,new_presents,new_futures = self.scene_mode.compute(inputs=new_input)
        for op in new_futures:
            text = "然后在"+str(op["hour"])+"点"+str(op["minute"])+"分"+mode_act_map[op["mode"]]+",到时候如果需要的话我在帮你"
            
            # 寻求用户的确认，如果返回确定的话就添加到time_depended_operation
            # 否则的话返回
            self.speak_queue.put(text)
            info = {
                "hour":op["hour"]
                ,"minute":op["minute"]
                ,"mode":op["mode"]
            }
        '''
        for op in pasts:
            text = "你刚刚在"+str(op["hour"])+"时"+str(op["minute"])+"分"+mode_act_map[op["mode"]]+"对吗"
            self.engine.say(text)
            self.engine.runAndWait()
            info = {
                "hour":op["hour"]
                ,"minute":op["minute"]
                ,"mode":op["mode"]
            }
            print(info)
        '''
        pass

    def device_predict(self,json_data):

        # 时间的信息已经藏在了场景模型里面了
        # 所以时间不在作为输入数据输入到模型的训练中

        if not self.can_self_control:
            return

        for device in self.devices.values():
            # 设备必须得经过训练才可以进行下面操作
            inputs = device.json_to_input(json_data = json_data)
            outputs = device.compute(inputs = inputs)
            # 发送给对应的下位机
            for output in outputs:
                output["code"] = "change_state"
                
                message = json.dumps(output)
                
                # 这里需要比较一下预测信息是否和现有的信息一致
                if device.state[output["data_name"]] is not output["data_value"]:
                    self.client.publish(self.pub_topic_to_device,message)
                    self.client.publish(self.pub_topic_change_device_property,message)
                    text = "已经把"+output["device_name"]+"的"+output["data_name"]+"调为"+str(output["data_value"])
                    self.speak_queue.put(text)
                    device.state[output["data_name"]] = output["data_value"]
                # 同时要把结果传给上位机，有上位机做判断
        pass

    def update_device_train_json(self,json_data):
        self.device_train_jsons.put(json_data)
        self.devices[json_data["device_name"]].append_train_data(json_data)
        pass

    def train_device(self):

        if not self.can_self_learn:
            return
        
        for device in self.devices.values():
            device.train_self()
            pass

        pass

    def auto_control(self):
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
                data = {
                    "year":year,
                    "month":month,
                    "day":day,
                    "hour":hour,
                    "minute":minute,
                    "mode":next_mode
                }
                self.change_mode(data)
                pass
        pass

    def speak_loop(self):

        # 创建一个语音引擎
        engine = pyttsx3.init()
        engine.setProperty('rate', 285)  # 设置语速，值越大语速越快

        def speak(text):
            engine.say(text)
            engine.runAndWait()
            pass

        while True:
            while not self.speak_queue.empty():
                text = self.speak_queue.get()
                print(text)
                speak(text=text)
                pass
            pass
    
    def enviro_loop(self):
        while True:
            data = self.weather.get_now()
            #print(data)
            time.sleep(10)
            pass
        pass

    def main_loop(self):
        count = 0
        while True:
            count = count + 0.01
            self.train_device()
            self.auto_control()
            while not self.predict_queue.empty():
                json_data = self.predict_queue.get()
                self.predict_next_scene(json_data)
                pass
            
            if count >=100:
                self.device_predict(self.current_enviro)
            pass
        pass

    def append_device(self,device_name,properties):
        if device_name in self.devices.keys():
            return
        text = "先生，正在尝试添加家居设备"+device_name+",过程会比较长"
        self.speak_queue.put(text)
        device = Device(device_name=device_name,device_properties=properties,mode_coder=self.mode_coder)
        self.devices[device_name] = device
        text = "添加家居设备"+device_name+"成功，我将会根据你的习惯指挥它工作"
        self.speak_queue.put(text)
        #print("添加新设备:",device_name,",属性:",properties)
        pass

    def change_mode(self,json_data):
        year = json_data["year"]
        month = json_data["month"]
        day = json_data["day"]
        hour = json_data["hour"]
        minute = json_data["minute"]
        mode = json_data["mode"]

        #print("current mode:",self.current_enviro["mode"],",next mode:",mode)
        if mode == self.current_enviro["mode"]:
            return False
        # 这个是预测的功能函数
        self.current_enviro["mode"] = mode
        text = mode_text[self.current_enviro["mode"]]
        self.speak_queue.put(text)
        
        #print(text)

        # 将切换场景的数据输入，让模型预测下一时刻的场景
        self.predict_queue.put(json_data)

        return True
        pass

    pass

    

core = Core()
core.run()