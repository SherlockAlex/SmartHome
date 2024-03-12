import requests
from bs4 import BeautifulSoup
import re

class Weather():
    def __init__(self):
        self.url = f"http://www.weather.com.cn/weather1d/101230508.shtml#search"
        pass
    
    def _parse_weather_info(self,weather_str):
        # 分割字符串
        parts = weather_str.split()

        # 提取各个部分
        week = parts[1]
        weather = parts[2]
        high_temp = int(parts[3].split('/')[0])
        low_temp = int(parts[3].split('/')[1][:-2])  # 去掉°C符号

        # 构建字典
        weather_info = {
            "week": week,
            "weather": weather,
            "daytime_temp": high_temp,
            "nighttime_temp": low_temp
        }

        return weather_info

    def get_now(self):
        response = requests.get(url=self.url)
        html_content = response.text
        soup = BeautifulSoup(html_content,"html.parser")
        
        info = soup.find("input",{"id":"hidden_title"})
        if info:
            value = info.get("value")
            value = value.encode("latin1").decode("utf-8")
            # 将其转为json

            data = self._parse_weather_info(value)
            data["code"] = 200
            return data
        else:
            data = {
                "code":408
            }
            return data
        
