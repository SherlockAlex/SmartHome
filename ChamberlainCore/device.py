from brain import Brain
import datatool as dt
import numpy as np

# 让设备自己学习再什么时间段做什么事，然后控制时间肯定不是每时每刻都在预测

class Device():
    def __init__(self,device_name,device_properties,mode_coder):
        self.__device_name = device_name
        self.properties = {}
        self.__device_properties = device_properties
        self.state = {}

        self.__train_data_file = "data_"+self.__device_name+".csv"
        self.__data_coder = dt.DataCoder()
        self.mode_coder = mode_coder

        self.x_self_train_list = []
        self.y_self_train_list = []

        
        select_columns=["mode","data_name","data_type","data_value"]
        feature_columns=["info_vx","info_vy","mode_vx","mode_vy","mode_vz","mode_vw","mode_va","mode_vb","mode_vc","mode_vd"]
        label_columns=["value_vx","value_vy"]
        self.__csv_data = dt.CSVData(
            filename=self.__train_data_file
            ,selected_columns=select_columns
            ,feature_columns=feature_columns
            ,label_columns=label_columns)
        
        self.__csv_data.set_prepare_callback(self.prepare)
        
        features_len = len(feature_columns)
        output_len = len(label_columns)
        self.__brain = Brain(
            input_dim=(features_len,)
            ,first_output_dim=features_len
            ,output_dim=output_len
        )

        for data_name,data_type in self.__device_properties.items():
            self.__data_coder.fit_transform(data_name=data_name,data_type=data_type)
            if data_type == "bool":
                self.state[data_name] = False
            elif data_type == "float":
                self.state[data_name] = 0.0

        try:
            self.load()
        except:
            pass

        pass
    
    def load(self):
        self.__brain.load(self.__device_name)

    def get_device_name(self):
        return self.__device_name
        pass

    def train(self):
        # 训练模型
        self.__csv_data.load_file()
        x_data,y_data = self.__csv_data.create_train_data()
        x_train,y_train,x_test,y_test = dt.slice_train_test_data(x_data=x_data,y_data=y_data,slice_rate=0.8)
        self.__brain.train(x_train,y_train,50,32,(x_test,y_test))
        self.__brain.save(self.__device_name)

        pass
    
    def append_train_data(self,json_data):
        data_info_vec = self.__data_coder.fit_transform(json_data["data_name"],json_data["data_type"])
        data_value_vec = self.__data_coder.transform(json_data["data_type"],json_data["data_value"])
        mode_vec = self.mode_coder.one_hot_transform(json_data["mode"])

        x_train = []
        for i in data_info_vec:
            x_train.append(i)
        for i in mode_vec:
            x_train.append(i)
        
        self.x_self_train_list.append(x_train)
        self.y_self_train_list.append(data_value_vec)

        pass

    def train_self(self):
        # 自学习
        
        if len(self.x_self_train_list)<=0 or len(self.y_self_train_list)<=0:
            self.x_self_train_list.clear()
            self.y_self_train_list.clear()
            return

        x = np.array(self.x_self_train_list)
        y = np.array(self.y_self_train_list)
        self.x_self_train_list.clear()
        self.y_self_train_list.clear()

        self.__brain.train(x,y,3,1,(x,y))
        self.__brain.save(self.__device_name)
        pass

    def test(self):
        #处理数据并获得df
        test_info1 = [2024,2,25,"sleep","switch","bool",-1]
        test_info2 = [2024,6,6,"wake_up","switch","bool",1]
        df = self.__csv_data.create_dataframe(
            [test_info1,test_info2])

        #将获得的数据加工出来，给神经网络预测
        input,y = self.__csv_data.create_data(df=df)
        output = self.compute(inputs=input)

        print(output)
        pass

    def json_to_input(self,json_data):
        mode = json_data["mode"]
        
        mode_vec = self.mode_coder.one_hot_transform(json_data["mode"])
        vector = []
        for i in mode_vec:
            vector.append(i)
        input = np.array([vector])
        return input
        pass

    def compute_attribute_state(self,inputs,data_name,data_type):
        # 首先计算switch的结果
        vec = self.__data_coder.fit_transform(data_name,data_type)
        input_list = []
        for i in vec:
            input_list.append(i)
        for i in inputs[0]:
            input_list.append(i)
        input = np.array([input_list])
        outputs = self.__brain.compute(inputs=input)
        return outputs[0]
        pass

    def compute(self,inputs):
        
        # 计算当前模式下可能的概率

        outputs = []
        for data_name,data_type in self.__device_properties.items():
            state_out = self.compute_attribute_state(inputs=inputs,data_name=data_name,data_type=data_type)
            state_value = self.__data_coder.inverse_transfrom(data_type,state_out)
            json_dict = {}
            json_dict["device_name"] = self.__device_name
            json_dict["data_name"] = data_name
            json_dict["data_type"] = data_type
            json_dict["data_value"] = state_value
            outputs.append(json_dict)
            pass
        
        return outputs
        pass

    def prepare(self,df):
        # 对df进行替换改造
        df["info_vx"],df["info_vy"] = zip(*df.apply(lambda row:self.__data_coder.fit_transform(row["data_name"],row["data_type"]),axis=1))
        
        df["value_vx"],df["value_vy"] = zip(*df.apply(lambda row:self.__data_coder.transform(row["data_type"],row["data_value"]),axis=1))
        df["mode_vx"],df["mode_vy"],df["mode_vz"],df["mode_vw"],df["mode_va"],df["mode_vb"],df["mode_vc"],df["mode_vd"] = zip(*df.apply(lambda row:self.mode_coder.one_hot_transform(row["mode"]),axis = 1))
        # 对设备信息进行编码
        df = df.drop(columns=["year","month","day","data_name","data_type","data_value"],axis = 1)
        print(df.head())
        return df
        pass
    
    pass