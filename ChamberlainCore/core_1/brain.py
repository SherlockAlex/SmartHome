import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import datatool as dt
import os
import math

class Brain():
    def __init__(self,device_coder,data_coder,input_dim,output_dim):
        self.device_coder = device_coder
        self.data_coder = data_coder
        self.model = tf.keras.Sequential()
        self.model.add(layers.Input(shape=input_dim))
        self.model.add(AttentionLayer(units=16))
        self.model.add(layers.Dense(32,activation = tf.nn.leaky_relu))
        self.model.add(layers.Dropout(0.3))
        self.model.add(layers.Dense(16,activation = tf.nn.leaky_relu))
        self.model.add(layers.Dropout(0.2))
        self.model.add(AttentionLayer(units=16))
        self.model.add(layers.Dense(16,activation = tf.nn.tanh))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(output_dim,activation = tf.nn.tanh))
        
        # 编译模型
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        pass

    def train(self,x_train,y_train,epochs,batch_size,validation_data):
        self.history = self.model.fit(
            x_train,y_train
            ,epochs=epochs
            ,batch_size = batch_size
            ,validation_data = validation_data)
        
        return self.history

        pass

    def compute(self,input):
        predict = self.model.predict(input)
        return predict
    
    def save(self,filename):
        self.model.save(filename)
        pass
        

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename,custom_objects={"AttentionLayer":AttentionLayer})

    pass

    def decode(self,outputs):
        result = []
        for output in outputs:
            #state = output[0]
            
            device = output[0:2]
            time = output[4:6]
            data = output[6:8]
            value = output[8:]

            device_info = self.device_coder.get_device(device)
            time_info = dt.vector_to_time(time)
            data_info = self.data_coder.get_data_info(data)
            data_value = self.data_coder.inverse_transfrom(type = data_info[1],vector = value)

            sequence = (device_info,time_info,data_info,data_value)
            result.append(sequence)
        return result

# 画曲线
def plot_performance(history=None,figure_directory=None,ylim_pad=[0,0]):
    xlabel="Epoch"
    legends=["Training","Validation"]
    
    plt.figure(figsize=(20,5))
    
    y1=history.history["accuracy"]
    y2=history.history["val_accuracy"]
    
    min_y=min(min(y1),min(y2))-ylim_pad[0]
    max_y=max(max(y1),max(y2))+ylim_pad[0]
    
    plt.subplot(121)
    
    plt.plot(y1)
    plt.plot(y2)
    
    plt.title("Model Accuracy\n",fontsize=17)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.ylim(min_y,max_y)
    plt.legend(legends,loc="upper left")
    plt.grid()
    
    y1=history.history["loss"]
    y2=history.history["val_loss"]
    
    min_y=min(min(y1),min(y2))-ylim_pad[1]
    max_y=max(max(y1),max(y2))+ylim_pad[1]
    
    plt.subplot(122)
    
    plt.plot(y1)
    plt.plot(y2)
    
    plt.title("Model Loss:\n",fontsize=17)
    plt.xlabel(xlabel,fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.ylim(min_y,max_y)
    plt.legend(legends,loc="upper left")
    plt.grid()
    plt.show()