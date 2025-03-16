import cv2
from loguru import logger
import layoutparser as lp
import cv2
import os
from pprint import pprint
from .segment_data import TextSegment
import numpy as np

class PDFImageProcessor:
    _model = None  # 静态类变量用于存储共享模型

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            # 确保模型缓存目录存在
            cache_dir = os.path.expanduser("~/.torch/iopath_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # model_name = "faster_rcnn_R_50_FPN_3x.pth"
            # model_path = os.path.join(os.path.dirname(__file__), "model", model_name)
            
            # os.makedirs(os.path.dirname(model_path), exist_ok=True)
            # 加载模型
            cls._model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
                label_map={0: "Text", 1: "Title",
                           2: "List", 3: "Table", 4: "Figure"},
                model_path='/home/ai-server/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth'
            )
            logger.info("Successfully loaded layout analysis model")
        return cls._model

    def __init__(self,
                 image_path: str,
                 segments_workspace: str,
                 image_enhance: bool = False,
                 max_width: int = 1200,
                 max_height: int = 336,
                 fix_size: bool = False,
                 fix_length: int = 672,
                 image_type: str = "research paper",
                #  ollama_ocr_config: dict = {}
                 ):
        self.image_path = image_path
        self.segments_workspace = segments_workspace
        self.max_width = max_width # 和模型输入的图片宽度一致
        self.image_enhance = image_enhance # 是否增强图片，根据图片类型决定
        self.max_height = max_height # 固定图片大小
        self.fix_size = fix_size # 是否固定图片大小
        self.fix_length = fix_length # 固定图片大小
        try:
            self.image = cv2.imread(self.image_path)
            # logger.info(
            #     f"Successfully read image with shape: {self.image.shape}")
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            raise e

        self.image_type = image_type
        if self.image_type == "research paper":
            self.model = self._load_model()
        else:
            self.model = None
        # logger.info(f"Image type set to: {self.image_type}")

        # self.ollama_ocr = OCRProcessor(**ollama_ocr_config)  # 初始化 ollama_ocr

        # 初始化 segments
        self.segments = []

    def pdf_image_process(self, extract_text: bool = True):
        if self.model is None:
            raise ValueError("布局分析模型未正确初始化，请检查模型加载日志")
        layout = self.model.detect(self.image)

        # print(layout)

        # 按照阅读顺序排序（先垂直后水平）
        layout.sort(key=lambda b: b.coordinates[1], inplace=True)

        # 处理每个布局区域
        # TODO:根据坐标的不连续性，判断是否有遗漏部分
        for idx, block in enumerate(layout):
            # 获取坐标并转换为整数（向下取整和向上取整以确保覆盖完整区域）
            # 左右扩充1个像素
            x_extend = 35
            x1 = max(0, int(block.block.x_1 - x_extend))
            y1 = max(0, int(block.block.y_1 - 9))
            x2 = min(self.image.shape[1], int(block.block.x_2) + x_extend)
            y2 = min(self.image.shape[0], int(block.block.y_2) + 7)

            # 裁剪图片
            segment = self.image[y1:y2, x1:x2]
            
            # 确定当前模块不是图，如果是图不处理
            if block.type != "Figure":
                # 图片最大像素,等比例缩放
                # 计算缩放比例，取宽高中较大的一边与max_width的比值

                if segment.shape[0] > self.fix_length and self.fix_size:
                    
                    scale = self.fix_length / segment.shape[0]  # 等比例缩放
                    segment = cv2.resize(segment, (self.fix_length, int(segment.shape[1] * scale)))
                else:
                    # 根据最大宽度，等比例缩放
                    height, width = segment.shape[:2]
                    if width > self.max_width or height > self.max_width:
                        scale = self.max_width / max(width, height)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        segment = cv2.resize(segment, (new_width, new_height))

            # 生成文件名：数字序号在前，类型在后
            segment_name = f"{idx:03d}_{block.type}.png"  # 修改文件名格式
            output_path = os.path.join(self.segments_workspace, segment_name)

            # 保存图片
            cv2.imwrite(output_path, segment)

            text_segment = TextSegment(
                image_path=output_path,
                type=block.type,
            )
            
            

            self.segments.append(text_segment)

        return layout
    
    def image_enhance_v1(self, image: np.ndarray):
        # Convert to grayscale if the image has multiple channels
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 图片锐化
        # equalized = cv2.equalizeHist(gray)
        
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # enhanced = clahe.apply(gray)
        
        #  # Denoise
        # denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # binary = cv2.adaptiveThreshold(
        #     denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #     cv2.THRESH_BINARY, 11, 2
        # )
        
        # # Sharpen the image to enhance text edges
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # sharpened = cv2.filter2D(binary, -1, kernel)
        
        return gray


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # from pprint import pprint
    image_processor = PDFImageProcessor(
        image_path="/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/images/page1.png",
        segments_workspace="/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page1",
        # ollama_ocr_config={
        #     "base_url": "http://192.168.5.72:11434/api/generate",}
    )
    layout = image_processor.pdf_image_process(extract_text=False)

    # 使用更简单的绘图方式，避免字体大小问题
    # image = lp.draw_box(image_processor.image,
    #                     layout,
    #                     box_width=3,
    #                     #    show_element_id=True
    #                     )

    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
