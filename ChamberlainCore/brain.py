import tensorflow as tf
from tensorflow.keras import layers
from nnlayers import AttentionLayer,SelfRecurrenceLayer
import datatool as dt

class Brain():
    def __init__(self,input_dim,output_dim):
        
        input_layer = layers.Input(shape = input_dim)
        dense1 = layers.Dense(32,activation = tf.nn.tanh)(input_layer)
        dropout1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(16,activation = tf.nn.tanh)(tf.concat([input_layer,dropout1],axis=1))
        dropout2 = layers.Dropout(0.2)(dense2)
        dense3 = layers.Dense(16, activation=tf.nn.tanh)(dropout2)
        dropout3 = layers.Dropout(0.2)(dense3)
        dense4 = layers.Dense(64,activation = tf.nn.sigmoid)(tf.concat([input_layer,dropout3],axis = 1))
        dense5 = layers.Dense(32,activation = tf.nn.sigmoid)(tf.concat([dropout2,dense4],axis=1))
        output = layers.Dense(output_dim,activation = tf.nn.tanh)(tf.concat([input_layer,dropout2,dropout3,dense4,dense5],axis = 1))

        self.model = tf.keras.Model(inputs = input_layer,outputs = output)

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

    def compute(self,inputs):
        predict = self.model.predict(inputs)
        return predict
    
    def save(self,filename):
        self.model.save(filename)
        pass
        

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename,custom_objects={
            "AttentionLayer":AttentionLayer
            ,"SelfRecurrenceLayer":SelfRecurrenceLayer
        })
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])
    pass

    pass
