from cnocr import CnOcr
import os
from loguru import logger
class OCREngine:
    _en_cn_ocr = None 
    _zh_cn_ocr = None  # 添加中文OCR引擎类变量
    
    @classmethod
    def _load_en_cn_ocr(cls):
        if cls._en_cn_ocr is None:
            cls._en_cn_ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
        return cls._en_cn_ocr
    
    @classmethod
    def _load_zh_cn_ocr(cls):  # 添加中文OCR引擎加载方法
        if cls._zh_cn_ocr is None:
            cls._zh_cn_ocr = CnOcr(det_model_name='zh_PP-OCRv3_det', rec_model_name='zh_PP-OCRv3')
        return cls._zh_cn_ocr
    
    def __init__(self):
        # 初始化OCR引擎
        # 为了加速，防止重复加载模型
        self.en_cn_ocr = None
        self.zh_cn_ocr = None
        
    def cn_ocr_image(self, image_path: str, ocr_type: str = "en")->list:
        # 使用cnocr作为引擎
        # 判断image_path是否存在
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
        # 如果ocr_type为en，则使用英文OCR引擎，否则使用中文OCR引擎
        if ocr_type == "en":
            self.en_cn_ocr = self._load_en_cn_ocr()
            result = self.en_cn_ocr.ocr(image_path)
        elif ocr_type == "zh":
            self.zh_cn_ocr = self._load_zh_cn_ocr()
            result = self.zh_cn_ocr.ocr(image_path)
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")
        
        # print("OCR result structure:", type(result))
        # if result:
        #     print("First item structure:", type(result[0]), result[0])
        # print(result)
        # 对result进行拼接
        result_str = ""
        for item in result:
            # CnOcr的返回结果格式是: 每个item包含位置信息[box]和文本内容[text]
            # 正确处理CnOcr返回的格式
            if isinstance(item, dict) and 'text' in item:
                result_str += item['text'] + " "
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                # 不同版本的CnOcr可能返回不同格式
                # 有些版本返回的是[box, text]格式
                if len(item) > 1:
                    result_str += item[1] + " "
                # 有些版本可能只返回text
                else:
                    result_str += str(item[0]) + " "
            else:
                # 直接转为字符串添加
                result_str += str(item) + " "
        
        return result_str.strip()

if __name__ == "__main__":
    ocr_engine = OCREngine()
    result = ocr_engine.cn_ocr_image(image_path="/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page7/006_Text.png")    
    print(type(result))
