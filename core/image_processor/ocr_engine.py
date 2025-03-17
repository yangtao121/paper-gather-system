from cnocr import CnOcr
import os
import base64
from loguru import logger
from core.dify import DifyClient
from core.config import DifyWorkflowConfig



class OCREngine:
    _en_cn_ocr = None 
    _zh_cn_ocr = None  # 添加中文OCR引擎类变量
    _dify_client = None
    
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
    
    @classmethod
    def _load_dify_client(cls, dify_workflow_config: DifyWorkflowConfig):
        if cls._dify_client is None:
            cls._dify_client = DifyClient(base_url=dify_workflow_config.api_url)
        return cls._dify_client
    
    def __init__(self, dify_workflow_config: DifyWorkflowConfig = None):
        # 初始化OCR引擎
        # 为了加速，防止重复加载模型
        self.en_cn_ocr = None
        self.zh_cn_ocr = None
        self.dify_workflow_config = dify_workflow_config
        if self.dify_workflow_config is not None:
            self._dify_client = self._load_dify_client(self.dify_workflow_config)
        
    def cn_ocr_image(self, image_path: str, ocr_type: str = "en",use_llm: bool = False)->str:
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
                
                
        ocr_result = result_str.strip()
        
        # 是否使用llm进行处理
        if use_llm:
            try:
                # 检查图片文件
                if not os.path.isfile(image_path):
                    logger.error(f"Image file not found or not accessible: {image_path}")
                    raise Exception(f"Image file not found or not accessible: {image_path}")
                
                file_size = os.path.getsize(image_path)
                # logger.info(f"Uploading image: {image_path}, size: {file_size} bytes")
                
                # 检查Dify客户端是否正确初始化
                if self._dify_client is None:
                    logger.error("Dify client not initialized")
                    raise Exception("Dify client not initialized")
                
                # 图片上传
                image_id = self._dify_client.upload_file(
                    file_path=image_path,
                    file_type='PNG',
                    user=self.dify_workflow_config.user,
                    mime_type='image/png',
                    api_key=self.dify_workflow_config.api_key
                )
                
                if image_id is None:
                    logger.error(f"Image upload failed: {image_path}")
                    raise Exception(f"Image upload failed: {image_path}")
                
                # logger.info(f"Image uploaded successfully with ID: {image_id}")
                
                # 调用Dify工作流程处理图片
                response = self._dify_client.workflow_run(
                    inputs={
                        "image": {
                            "transfer_method": "local_file",
                            "upload_file_id": image_id,
                            "type": "image"
                        },
                        "ocr_result": ocr_result
                    },
                    api_key=self.dify_workflow_config.api_key,
                    user=self.dify_workflow_config.user
                )
                
                print(response)
                
                # 根据Dify客户端的返回格式获取结果
                # 注意：可能需要根据实际的DifyClient实现调整这部分代码
                if isinstance(response, dict) and "result" in response:
                    result_str = response["result"]
                else:
                    result_str = str(response)
                
                # print(response)
                
                try:
                    llm_eval = eval(response['data']['outputs']['output'])
                except Exception as e:
                    # logger.warning(f"Error during LLM processing: {str(e)}")
                    return ocr_result
                # print(llm_eval)
                
                # print(llm_eval['content'])
                
                return llm_eval['content']
            
            except Exception as e:
                logger.exception(f"Error during LLM processing: {str(e)}")
                # 如果LLM处理失败，返回原始OCR结果
                return ocr_result
        
        
        return result_str

if __name__ == "__main__":
    ocr_engine = OCREngine()
    result = ocr_engine.cn_ocr_image(image_path="/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page2/005_Text.png")    
    print(type(result))
