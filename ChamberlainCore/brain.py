import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from nnlayers import AttentionLayer,SelfRecurrenceLayer,FeedbackDense
import datatool as dt
import numpy as np

class TimeTraveler():

    def Significance(self,units,input_layer):
        key = layers.Dense(units,activation=None)(input_layer)
        query = layers.Dense(units,activation=None)(input_layer)
        value = layers.Dense(units,activation=None)(input_layer)
        mask = layers.Multiply()([key,query])
        mask = layers.Softmax()(mask)
        output = layers.Multiply()([value,mask])
        output = layers.Add()([output,value])
        return output
        pass

    def self_encoder_layer(self,input_layer):
        
        significance_layers = []
        for i in range(3):
            significance = self.Significance(8,input_layer)
            significance_layers.append(significance)
        significance = layers.Average()(significance_layers)


        hidden_layer1 = layers.Dense(32, activation=tf.nn.tanh)(significance)
        h_dropout1 = layers.Dropout(0.3)(hidden_layer1)
        hidden_layer2 = layers.Dense(64, activation=tf.nn.tanh)(h_dropout1)
        h_dropout2 = layers.Dropout(0.3)(hidden_layer2)
        hidden_layer = layers.Dense(128, activation=tf.nn.tanh)(h_dropout2)
        h_dropout = layers.Dropout(0.3)(hidden_layer)
        return h_dropout
        pass

    def decoder_layer(self,input_layer):
        inv_hidden_layer2 = layers.Dense(64, activation=tf.nn.tanh)(input_layer)
        inv_h_dropout2 = layers.Dropout(0.3)(inv_hidden_layer2)
        inv_hidden_layer1 = layers.Dense(32, activation=tf.nn.tanh)(inv_h_dropout2)
        inv_h_dropout1 = layers.Dropout(0.3)(inv_hidden_layer1)
        return inv_h_dropout1
        pass

    def dual_layer(self,input_layer,present_dims,past_dims,future_dims):
        
        #预测现在(自编码器层)
        
        present_encoder_out = self.self_encoder_layer(input_layer)
        decoder_output = self.decoder_layer(present_encoder_out)

        present_output = layers.Dense(present_dims, activation=tf.nn.tanh)(decoder_output)

        #回溯过去(现在信息)
        past_significance_layers = []
        for i in range(8):
            significance = self.Significance(8,present_encoder_out)
            past_significance_layers.append(significance)
        past_significance = layers.Average()(past_significance_layers)

        
        past_dense1 = layers.Dense(32,activation = tf.nn.tanh)(past_significance)
        past_dropout1 = layers.Dropout(0.3)(past_dense1)
        past_link1 = tf.concat([past_dropout1,past_significance],axis = 1)
        past_dense2 = layers.Dense(64,activation = tf.nn.tanh)(past_link1)
        past_dropout2 = layers.Dropout(0.3)(past_dense2)
        past_link2 = tf.concat([past_dropout2,past_dropout1],axis=1)
        past_dense3 = layers.Dense(128,activation = tf.nn.tanh)(past_link2)
        past_dropout3 = layers.Dropout(0.3)(past_dense3)
        past_link3 = tf.concat([past_dropout3,past_dropout1],axis=1)
        past_dense4 = layers.Dense(64,activation = tf.nn.tanh)(past_link3)
        past_dropout4 = layers.Dropout(0.3)(past_dense4)
        past_link4 = tf.concat([past_dropout4,past_significance],axis = 1)
        past_output = layers.Dense(past_dims, activation=tf.nn.tanh)(past_link4)

        #预测未来(现在信息,过去信息(记忆))
        past_present_info = layers.Concatenate()([present_encoder_out,past_dropout3])
        
        future_significance_layers = []
        for i in range(8):
            significance = self.Significance(16,past_present_info)
            future_significance_layers.append(significance)
        future_significance = layers.Average()(future_significance_layers)

        future_dense1 = layers.Dense(32,activation = tf.nn.tanh)(future_significance)
        future_dropout1 = layers.Dropout(0.3)(future_dense1)
        future_link1 = tf.concat([future_dropout1,future_significance],axis = 1)
        future_dense2 = layers.Dense(64,activation = tf.nn.tanh)(future_link1)
        future_dropout2 = layers.Dropout(0.3)(future_dense2)
        future_link2 = tf.concat([future_dropout2,future_dropout1],axis=1)
        future_dense3 = layers.Dense(32,activation = tf.nn.tanh)(future_link2)
        future_dropout3 = layers.Dropout(0.3)(future_dense3)
        future_link3 = tf.concat([future_dropout3,future_dropout1],axis=1)
        future_dense4 = layers.Dense(32,activation = tf.nn.tanh)(future_link3)
        future_dropout4 = layers.Dropout(0.3)(future_dense4)
        future_link4 = tf.concat([future_dropout4,future_significance],axis = 1)
        future_output = layers.Dense(future_dims, activation=tf.nn.tanh)(future_link4)
    
        # 最后output2的结果还要经过其他编码器进行解码
        return past_output,present_output, future_output
    
    def __init__(self,input_shape,present_dims,past_dims,future_dims):
        input_layer = layers.Input(shape=input_shape)
        past1,present1, future1 = self.dual_layer(input_layer=input_layer,present_dims=present_dims,past_dims=past_dims,future_dims=future_dims)
        past2,present2, future2 = self.dual_layer(input_layer=present1,present_dims=present_dims,past_dims=past_dims,future_dims=future_dims)
        past3,present3,future3 = self.dual_layer(input_layer=present2,present_dims=present_dims,past_dims=past_dims,future_dims=future_dims)
        
        past = layers.Average()([past1,past2,past3])
        present = layers.Average()([present1,present2,present3])
        future = layers.Average()([future1,future2,future3])
        
        self.model = tf.keras.Model(inputs = input_layer,outputs=[present,future,past])
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        # 打印模型结构
        self.model.summary()
        pass

    def train(self,present_train,past_train,future_train,epochs,batch_size,validation_data):
        
        log_dir = "logs/fit/"
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        

        self.train_history = self.model.fit(
            present_train
            ,[present_train,future_train,past_train]
            ,epochs=epochs
            ,callbacks=[tensorboard_callback]
            ,batch_size = batch_size
            ,validation_data = validation_data)
        
        return self.train_history

        pass
    

    def compute(self,inputs):
        present,future,past= self.model.predict(inputs,verbose=0)
        return past,present,future
    
    def save(self,filename):
        self.model.save(filename)
        pass
        

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        pass
    pass



class DualBrain():

    def Significance(self,units,input_layer):
        key = layers.Dense(units,activation=None)(input_layer)
        query = layers.Dense(units,activation=None)(input_layer)
        value = layers.Dense(units,activation=None)(input_layer)
        mask = layers.Multiply()([key,query])
        mask = layers.Softmax()(mask)
        output = layers.Multiply()([value,mask])
        output = layers.Add()([output,value])
        return output
        pass

    def self_encoder(self,input_layer):
        
        significance_layer = []
        for i in range(3):
            significance = self.Significance(6,input_layer)
            significance_layer.append(significance)
            pass
        significance = layers.Average()(significance_layer)
        
        hidden_layer1 = layers.Dense(64, activation=tf.nn.tanh)(significance)
        h_dropout1 = layers.Dropout(0.3)(hidden_layer1)
        hidden_layer2 = layers.Dense(128, activation=tf.nn.tanh)(h_dropout1)
        h_dropout2 = layers.Dropout(0.3)(hidden_layer2)
        hidden_layer = layers.Dense(256, activation=tf.nn.tanh)(h_dropout2)
        h_dropout = layers.Dropout(0.3)(hidden_layer)
        return h_dropout 
        pass
    
    def self_decoder(self,input_layer):
        inv_hidden_layer2 = layers.Dense(128, activation=tf.nn.tanh)(input_layer)
        inv_h_dropout2 = layers.Dropout(0.3)(inv_hidden_layer2)
        inv_hidden_layer1 = layers.Dense(64, activation=tf.nn.tanh)(inv_h_dropout2)
        inv_h_dropout1 = layers.Dropout(0.3)(inv_hidden_layer1)
        return inv_h_dropout1
        pass

    def dual_output_nn(self,input_layer,first_output_dim, output_units):
        

        #(自编码器)
        self_encoder_out = self.self_encoder(input_layer=input_layer)              # 代表着Brain对当前状态的信息，同时也要存放在经验吃池中
        self_decoder_out = self.self_decoder(self_encoder_out)

        self_out = layers.Dense(first_output_dim, activation=tf.nn.tanh)(self_decoder_out)

        #(预测信息)
        dense1 = layers.Dense(32,activation = tf.nn.tanh)(self_encoder_out)
        dropout1 = layers.Dropout(0.3)(dense1)
        link1 = tf.concat([dropout1,self_encoder_out],axis = 1)
        dense2 = layers.Dense(32,activation = tf.nn.tanh)(link1)
        dropout2 = layers.Dropout(0.3)(dense2)
        link2 = tf.concat([dropout2,dropout1],axis=1)
        dense3 = layers.Dense(16,activation = tf.nn.tanh)(link2)
        dropout3 = layers.Dropout(0.3)(dense3)
        link3 = tf.concat([dropout3,dropout1],axis=1)
        dense4 = layers.Dense(16,activation = tf.nn.tanh)(link3)
        dropout4 = layers.Dropout(0.3)(dense4)
        link4 = tf.concat([dropout4,self_encoder_out],axis = 1)
        predict_out = layers.Dense(output_units, activation=tf.nn.tanh)(link4)
        
        
        return self_out,predict_out

    def __init__(self,input_dim,first_output_dim,output_dim):
        
        input_layer = layers.Input(shape=input_dim)
        
        self_out1,predict_out1 = self.dual_output_nn(input_layer=input_layer,first_output_dim=first_output_dim,output_units=output_dim)
        self_out2,predict_out2 = self.dual_output_nn(input_layer=self_out1,first_output_dim=first_output_dim,output_units=output_dim)
        self_out3,predict_out3 = self.dual_output_nn(input_layer=self_out2,first_output_dim=first_output_dim,output_units=output_dim)

        self_out = tf.keras.layers.Average()([self_out1,self_out2,self_out3])
        predict_out = tf.keras.layers.Average()([predict_out1,predict_out2,predict_out3])

        self.model = tf.keras.Model(inputs=input_layer, outputs=[self_out, predict_out])
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        # 打印模型结构
        #self.model.summary()
        pass

    def train(self,x_train,y_train,epochs,batch_size,validation_data):
        self.history = self.model.fit(
            x_train
            ,[x_train,y_train]
            ,epochs=epochs
            ,batch_size = batch_size
            ,validation_data = (validation_data[0],[validation_data[0],validation_data[1]]))
        
        return self.history

        pass

    def train_dual(self,x_train,y_train,z_train,epochs,batch_size,validation_data):
        self.history = self.model.fit(
            x_train
            ,[y_train,z_train]
            ,epochs = epochs
            ,batch_size = batch_size
            ,validation_data = validation_data
        )
        return self.history
        pass
    

    def compute(self,inputs):
        x_encoder_decode, predict= self.model.predict(inputs,verbose=0)
        return predict
    
    def save(self,filename):
        self.model.save(filename)
        pass
        

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
    pass

class Brain():

    def Significance(self,units,input_layer):
        key = layers.Dense(units,activation=None)(input_layer)
        query = layers.Dense(units,activation=None)(input_layer)
        value = layers.Dense(units,activation=None)(input_layer)
        mask = layers.Multiply()([key,query])
        mask = layers.Softmax()(mask)
        output = layers.Multiply()([value,mask])
        output = layers.Add()([output,value])
        return output
        pass

    def res_net(self,input_layer,output_dims):
        dense1 = layers.Dense(16,activation=tf.nn.tanh)(input_layer)
        dropout1 = layers.Dropout(0.3)(dense1)
        link1 = tf.concat([dropout1,input_layer],axis = 1)
        dense2 = layers.Dense(16,activation=tf.nn.tanh)(link1)
        dropout2 = layers.Dropout(0.3)(dense2)
        link2 = tf.concat([dropout2,input_layer],axis = 1)
        dense3 = layers.Dense(16,activation=tf.nn.tanh)(link2)
        dropout3 = layers.Dropout(0.3)(dense3)
        link3 = tf.concat([dropout3,input_layer],axis = 1)
        output = layers.Dense(output_dims,activation = tf.nn.tanh)(link3)
        return output
        pass

    def __init__(self,input_dim,first_output_dim,output_dim):
        
        input_layer = layers.Input(shape=input_dim)
        
        significance_layers = []
        for i in range(3):
            significance_layer = self.Significance(8,input_layer=input_layer)
            significance_layers.append(significance_layer)
            pass
        significance = layers.Average()(significance_layers)
        output = self.res_net(significance,output_dims=output_dim)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
        # 打印模型结构
        #self.model.summary()
        pass

    def train(self,x_train,y_train,epochs,batch_size,validation_data):
        self.history = self.model.fit(
            x_train
            ,y_train
            ,epochs=epochs
            ,batch_size = batch_size
            ,validation_data = (validation_data[0],validation_data[1]))
        
        return self.history

        pass
    

    def compute(self,inputs):
        predict= self.model.predict(inputs,verbose=0)
        return predict
    
    def save(self,filename):
        self.model.save(filename)
        pass
        

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
    pass

    pass

class ConceptionVector():
    def __init__(self,vector_dims,map) -> None:
        self.map = map
        self.one_hot_dims = len(map)
        input_layer = layers.Input(self.one_hot_dims)
        vector_layer = layers.Dense(vector_dims,activation = tf.nn.tanh)(input_layer)
        output = layers.Dense(self.one_hot_dims,activation =tf.nn.softmax)(vector_layer)
        self.one_hot_model = tf.keras.Model(inputs = input_layer,outputs = output)
        # 编译模型
        self.one_hot_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        
        x = []
        y = []
        for i in range(1000):
            for key,value in map.items():
                vector = np.zeros(self.one_hot_dims)
                vector[value] = 1
                y.append(vector)
                x.append(vector)
        x = np.array(x)
        y = np.array(y)

        self.one_hot_model.fit(x,y,epochs=3)
        self.embedding = {}
        self.model = tf.keras.Model(inputs = self.one_hot_model.input,outputs=self.one_hot_model.layers[1].output)
        
        for key in map.keys():
            self.get_vector(key)

        pass

    def get_vector(self,input):
        i = np.zeros(self.one_hot_dims)
        code_value = self.map[input]
        i[code_value] = 1
        in_vec = np.array([i])
        outputs= self.model.predict(in_vec)
        self.embedding[input] = outputs[0]
        return outputs[0]
        pass

    def get_conception(self,vector):
        out = []
        for key,vec in self.embedding.items():
            dot = np.dot(vec,vector)
            norm_a = np.linalg.norm(vector)
            norm_b = np.linalg.norm(vec)
            sim = dot/(norm_a*norm_b)
            out.append((key,sim))
        return out

    pass

class Variable2Vec():
    def __init__(self,variable,type_map):
        
        pass
    pass
