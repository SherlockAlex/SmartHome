import tensorflow as tf

class SelfRecurrenceLayer(tf.keras.layers.Layer):
    def __init__(self,units,activation,name="self_recurrence",trainable=True,dtype=tf.float32):
        super(SelfRecurrenceLayer,self).__init__()
        self.units = units
        self.activation = activation
        self.memory = tf.Variable(tf.zeros(shape=(1,units)))
        pass

    def build(self,input_shape):
        self.kernel = self.add_weight("kernel", shape=(input_shape[-1], self.units),initializer="random_normal",trainable = True)
        self.self_kernel = self.add_weight("self_kernel", shape=(self.units, self.units),initializer="random_normal",trainable = True)
        
        pass

    def call(self,inputs):
        x1 = tf.matmul(inputs,self.kernel)
        print(x1)
        x2 = tf.matmul(self.self_kernel,tf.expand_dims(self.memory,axis=1))
        x = tf.add(x1,x2)
        output = self.activation(x)
        self.memory.assign(output)
        return output
        pass

    def get_config(self):
        config = {
            "units":self.units
            ,"activation":self.activation
        }
        base_config = super(AttentionLayer,self).get_config()
        return dict(list(base_config.items())+list(config.items()))
        pass

    pass

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self,units,name="attention",trainable=True,dtype = tf.float32):
        super(AttentionLayer,self).__init__()
        self.units = units
        pass

    def build(self,input_shape):
        # 构建参数
        self.Wq = self.add_weight(name = "query_matrix",shape=(input_shape[-1],self.units),initializer="random_normal",trainable=True)
        self.Wk = self.add_weight(name = "key_matrix",shape=(input_shape[-1],self.units),initializer = "random_normal",trainable=True)
        self.Wv = self.add_weight(name="value_matrix",shape=(input_shape[-1],self.units),initializer="random_normal",trainable=True)

        pass

    def call(self,inputs):
        Q = tf.matmul(inputs,self.Wq)
        K = tf.matmul(inputs,self.Wk)
        V = tf.matmul(inputs,self.Wv)

        # 然后Q，K进行内积
        attention_weight = tf.nn.softmax(tf.matmul(Q,V,transpose_b=True))
        outputs = tf.matmul(attention_weight,V)
        return outputs

        pass

    def get_config(self):
        config = {"units":self.units}
        base_config = super(AttentionLayer,self).get_config()
        return dict(list(base_config.items())+list(config.items()))
        pass

    pass