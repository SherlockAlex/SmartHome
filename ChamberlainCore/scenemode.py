from brain import DualBrain
from brain import TimeTraveler
import datatool as dt
import numpy as np
import tensorflow as tf

class SceneMode():
    def __init__(self,mode_coder):
        columns = ["year","month","day","hour","minute","mode"]
        self.csv_data = dt.CSVDualData("./data_mode.csv",columns=columns,step=1)
        self.csv_data.set_encode_callback(self.csv_encode)
        self.csv_data.set_preprocess_callback(self.prepare)
        self.csv_data.load_file()

        self.mode_coder = mode_coder

        self.net = TimeTraveler((12,),12,12,12)

        pass

    def json_to_input(self,json_data):
        year = json_data["year"]
        month = json_data["month"]
        day = json_data["day"]
        hour = json_data["hour"]
        minute = json_data["minute"]
        mode = json_data["mode"]
        date_vec = dt.date_to_vector(year=year,month=month,day=day)
        time_vec = dt.time_to_vector(hour=hour,minute=minute)
        mode_vec = self.mode_coder.one_hot_transform(mode=mode)
        vector = []
        for i in date_vec:
            vector.append(i)
        for i in time_vec:
            vector.append(i)
        for i in mode_vec:
            vector.append(i)
        input = np.array([vector])
        return input
        pass
    
    def compute(self,inputs):
        past_out,present_out,future_out = self.net.compute(inputs = inputs)
        past = self.__decode(outputs=past_out)
        present = self.__decode(outputs=present_out)
        future = self.__decode(outputs=future_out)
        return past,present,future
        pass

    def train_self(self,present_json,future_json,past_json):

        present = self.json_to_input(present_json)
        future = self.json_to_input(future_json)
        past = self.json_to_input(past_json)

        self.net.train(present_train=present,past_train=past,future_train=future,epochs=1,batch_size=1,validation_data=(present,[present,future,past]))
        self.save()
        pass

    def train(self):
        self.csv_data.load_file()
        present_data,future_data,past_data = self.csv_data.create_train_sequence()
        present_data = present_data.reshape((present_data.shape[0],present_data.shape[2]))
        past_data = past_data.reshape((past_data.shape[0],past_data.shape[2]))
        present_train,futrue_train,past_train,present_test,future_test,past_test = dt.slice_dual_train_test_data(x_data=present_data,y_data=future_data,z_data=past_data,slice_rate=0.8)
        self.net.train(present_train=present_train,past_train=past_train,future_train=futrue_train,epochs=30,batch_size=32,validation_data=(present_test,[present_test,future_test,past_test]))
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
        df["date_vx"],df["date_vy"] = zip(*df.apply(lambda row:dt.date_to_vector(year=row["year"],month=row["month"],day=row["day"]),axis=1))
        df["time_vx"],df["time_vy"] = zip(*df.apply(lambda row:dt.time_to_vector(hour=row["hour"],minute=row["minute"]),axis = 1))
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
            date_vec = output[0:2]
            time_vec = output[2:4]
            mode_vec = output[4:]
            
            next_date = dt.vector_to_date(2024,date_vec)
            next_time = dt.vector_to_time(time_vec)
            next_mode = self.mode_coder.get_mode(mode_vec)
            
            json_dict = {}
            json_dict["year"] = 2024
            json_dict["month"] = next_date[0]
            json_dict["day"] = next_date[1]
            json_dict["hour"] = next_time[0]
            json_dict["minute"] = next_time[1]
            json_dict["mode"] = next_mode
            results.append(json_dict)
        return results

    pass