import easyocr

# 创建 OCR 阅读器
reader = easyocr.Reader(['en', 'ch_sim'])  # 支持英文和简体中文

# 识别图片中的文本
result = reader.readtext(
    '/Users/yangtao/Documents/code.nosync/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page4/005_Text.pngpip install ollama-ocr'
)
for detection in result:
    print(detection[1])  # 打印识别到的文本
