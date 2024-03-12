from brain import Brain
import datatool as dt
import numpy as np

class SceneMode():
    def __init__(self,mode_coder):
        columns = ["year","month","day","hour","minute","mode"]
        self.csv_data = dt.CSVSequnceData("./data_mode.csv",columns=columns,step=1)
        self.csv_data.set_encode_callback(self.csv_encode)
        self.csv_data.set_preprocess_callback(self.prepare)
        self.csv_data.load_file()

        self.mode_coder = mode_coder

        self.net = Brain(
            input_dim=10
            ,output_dim=10
        )

        pass

    def json_to_input(self,json_data):
        hour = json_data["hour"]
        minute = json_data["minute"]
        mode = json_data["mode"]
        time_vec = dt.time_to_vector(hour=hour,minute=minute)
        mode_vec = self.mode_coder.one_hot_transform(mode=mode)
        vector = []
        for i in time_vec:
            vector.append(i)
        for i in mode_vec:
            vector.append(i)
        input = np.array([vector])
        return input
        pass
    
    def compute(self,inputs):
        out_vecs = self.net.compute(inputs = inputs)
        outputs = self.__decode(outputs=out_vecs)
        return outputs
        pass

    def train_self(self,x_json,y_json):
        # 理论上，应该是用户操作四次后进行学习
        # 前三次作为输入，进行加权求和，记录越靠后的，权值越大

        # 但是这里我们让
        x = self.json_to_input(x_json)
        y = self.json_to_input(y_json)
        self.net.train(x,y,1,32,(x,y))
        self.save()
        pass

    def train(self):
        self.csv_data.load_file()
        x_data,y_data = self.csv_data.create_train_sequence()
        x_data = x_data.reshape((x_data.shape[0],x_data.shape[2]))
        x_train,y_train,x_test,y_test = dt.slice_train_test_data(x_data=x_data,y_data=y_data,slice_rate=0.8)
        self.net.train(x_train,y_train,15,32,(x_test,y_test))
        self.save()
        pass

    def test(self):
        #处理数据并获得df
        test_info1 = [2024,2,25,18,3,"homecoming"]
        test_info2 = [2024,6,6,18,4,"homecoming"]
        test_info3 = [2024,6,6,7,3,"morning_dining"]
        test_info4 = [2024,6,6,22,5,"sleep"]
        test_info5 = [2024,6,6,0,56,"sleep"]
        test_info6 = [2024,6,6,6,3,"wake_up"]
        test_info7 = [2024,6,6,6,45,"morning_dining"]
        df = self.csv_data.create_dataframe(
            [test_info1,test_info2,test_info3,test_info4,test_info5,test_info6,test_info7])

        #将获得的数据加工出来，给神经网络预测
        input = self.csv_data.create_sequence(df=df)
        input = input.reshape(input.shape[0],input.shape[2])
        output = self.compute(inputs=input)
        for sequence in output:
            print(sequence)
        pass

    def save(self):
        self.net.save("javis")
        pass

    def load(self):
        self.net.load("javis")
        pass

    def prepare(self,df):
        # 对df进行替换改造
        #df["date_vx"],df["date_vy"] = zip(*df.apply(lambda row:dt.date_to_vector(year=row["year"],month=row["month"],day=row["day"]),axis=1))
        df["time_vx"],df["time_vy"] = zip(*df.apply(lambda row:dt.time_to_vector(hour=row["hour"],minute=row["minute"]),axis = 1))
        #df["mode_vx"],df["mode_vy"],df["mode_vz"] = zip(*df.apply(lambda row:self.mode_coder.fit_transform(mode=row["mode"]),axis = 1))
        df["mode_vx"],df["mode_vy"],df["mode_vz"],df["mode_vw"],df["mode_va"],df["mode_vb"],df["mode_vc"],df["mode_vd"] = zip(*df.apply(lambda row:self.mode_coder.one_hot_transform(mode=row["mode"]),axis = 1))
        df = df.drop(columns=["year","month","day","hour","minute","mode"],axis = 1)
        return df
    pass

    def csv_encode(self,dataset):
        return dataset
    
    def __decode(self,outputs):
        # 编码器的功能
        results = []
        for output in outputs:
            time_vec = output[0:2]
            mode_vec = output[2:]
            next_time = dt.vector_to_time(time_vec)
            next_mode = self.mode_coder.get_mode(mode_vec)
            json_dict = {}
            json_dict["hour"] = next_time[0]
            json_dict["minute"] = next_time[1]
            json_dict["mode"] = next_mode
            results.append(json_dict)
        return results

    pass