from core.image_processor.ocr_engine import OCREngine
from core.config import DifyWorkflowConfig
from loguru import logger
import pprint

if __name__ == "__main__":
    try:
        dify_workflow_config = DifyWorkflowConfig(
            api_url="http://192.168.5.54:5001",
            api_key="app-ZwA8BICe2IJqIFSVMGWTmPNC",
            user="user123"
        )
        # logger.info(f"Testing with config: {dify_workflow_config}")
        
        ocr_engine = OCREngine(dify_workflow_config=dify_workflow_config)
        
        image_path = "/home/ai-server/dev/paper-gather-system/src/papers/nomad/segments/page4/016_Title.png"
        # logger.info(f"Processing image: {image_path}")
        
        # # 先测试不使用LLM
        result_no_llm = ocr_engine.cn_ocr_image(image_path=image_path, use_llm=False)
        print("result_no_llm########################")
        print(result_no_llm)
        
        # 再测试使用LLM
        result = ocr_engine.cn_ocr_image(image_path=image_path, use_llm=True)
        # logger.info(f"OCR result with LLM: {result[:100]}...")
        print("after LLM result########################")
        print(result)

        
        
        
    except Exception as e:
        logger.exception(f"Test failed: {str(e)}")
