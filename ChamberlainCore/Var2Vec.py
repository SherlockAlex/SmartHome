import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import hashlib
from math import sin,cos

def fixed_hash(s):
    sha256 = hashlib.sha256()
    sha256.update(s.encode('utf-8'))
    hash_bytes = sha256.digest()
    hash_int = int.from_bytes(hash_bytes,byteorder="big")
    return hash_int

class Var2Vec():
    def __init__(self,vector_dims):
        # 利用变量名和变量类型预测变量类型
        # float bool
        input_layer = layers.Input(3)
        vector_layer = layers.Dense(vector_dims,activation = tf.nn.tanh)(input_layer)
        type_layer = layers.Dense(2,activation = tf.nn.softmax)(vector_layer)

        self.code_model = tf.keras.Model(inputs = input_layer,outputs = type_layer)
        self.model = tf.keras.Model(inputs = self.code_model.input,outputs=self.code_model.layers[1].output)
        self.code_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        self.embedding = {}
        pass

    def create_data(self,data):
        y_data = []
        x_data = []
        for key,value in data.items():
            type_vec = [0,0]
            if value == "bool":
                code = 0
            elif value == "float":
                code = 1
            type_vec[code] = 1
            y_data.append(type_vec)

            name_radian = fixed_hash(key)
            name_code = sin(name_radian)
            x = []
            x.append(name_code)
            for i in type_vec:
                x.append(i)

            x_data.append(x)

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data,y_data
        pass

    def train(self,data,epochs):
        for i in range(100):
            x_data,y_data = self.create_data(data)
            self.code_model.fit(x_data,y_data,epochs)
        self.vv(data)
        pass
    
    def vv(self,maps):
        x_data,y_data = self.create_data(maps)

        outputs = self.model.predict(x_data)
        i = 0
        for key,value in maps.items():
            self.embedding[(key,value)] = outputs[i]
            i = i+1
            pass
        return outputs

        pass

    def similary(self,vector):
        out = []
        for key,vec in self.embedding.items():
            dot = np.dot(vec,vector)
            norm_a = np.linalg.norm(vector)
            norm_b = np.linalg.norm(vec)
            sim = dot/(norm_a*norm_b)
            out.append((key,sim))
        return out
        pass

    pass



model = Var2Vec(3)
model.train({
    "switch":"bool"
    ,"left_motor":"float"
    ,"right_motor":"float"
    ,"temp":"float"
    ,"forward":"bool"
},1000)

o = model.vv({"switch1":"bool"})
similary = model.similary(o[0])
print(o,similary)