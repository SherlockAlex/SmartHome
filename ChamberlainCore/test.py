from pocketsphinx import LiveSpeech

keywords = ["mute"]
# 创建一个实时语音识别器
speech = LiveSpeech(lm=True, kws_threshold=1e-20,keywords = keywords)

print("开始语音识别...")
for phrase in speech:
    print(phrase)