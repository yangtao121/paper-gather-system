import cv2
from loguru import logger
import layoutparser as lp
import cv2
import os
from pprint import pprint


class ImageProcessor:
    _model = None  # 静态类变量用于存储共享模型

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            # 确保模型缓存目录存在
            cache_dir = os.path.expanduser("~/.torch/iopath_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # 加载模型
            cls._model = lp.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0: "Text", 1: "Title",
                           2: "List", 3: "Table", 4: "Figure"}
            )
            logger.info("Successfully loaded layout analysis model")
        return cls._model

    def __init__(self,
                 image_path: str,
                 segments_workspace: str,
                 image_type: str = "research paper"):
        self.image_path = image_path
        self.segments_workspace = segments_workspace

        try:
            self.image = cv2.imread(self.image_path)
            logger.info(
                f"Successfully read image with shape: {self.image.shape}")
        except Exception as e:
            logger.error(f"Failed to read image: {e}")
            raise e

        self.image_type = image_type
        if self.image_type == "research paper":
            self.model = self._load_model()
        else:
            self.model = None
        logger.info(f"Image type set to: {self.image_type}")

    def pdf_image_process(self):
        if self.model is None:
            raise ValueError("布局分析模型未正确初始化，请检查模型加载日志")
        layout = self.model.detect(self.image)

        print(layout)

        # 按照阅读顺序排序（先垂直后水平）
        layout.sort(key=lambda b: b.coordinates[1], inplace=True)

        # 处理每个布局区域
        for idx, block in enumerate(layout):
            # 获取坐标并转换为整数（向下取整和向上取整以确保覆盖完整区域）
            # 左右扩充1个像素
            x_extend = 35
            x1 = max(0, int(block.block.x_1 - x_extend))
            y1 = max(0, int(block.block.y_1 - 10))
            x2 = min(self.image.shape[1], int(block.block.x_2) + x_extend)
            y2 = min(self.image.shape[0], int(block.block.y_2) + 20)

            # 裁剪图片
            segment = self.image[y1:y2, x1:x2]

            # 生成文件名：数字序号在前，类型在后
            segment_name = f"{idx:03d}_{block.type}.png"  # 修改文件名格式
            output_path = os.path.join(self.segments_workspace, segment_name)

            # 保存图片
            cv2.imwrite(output_path, segment)

        return layout


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # from pprint import pprint
    image_processor = ImageProcessor(
        image_path="/Users/yangtao/Documents/code.nosync/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/images/page1.png",
        segments_workspace="/Users/yangtao/Documents/code.nosync/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page1"
    )
    layout = image_processor.pdf_image_process()

    # 使用更简单的绘图方式，避免字体大小问题
    # image = lp.draw_box(image_processor.image,
    #                     layout,
    #                     box_width=3,
    #                     #    show_element_id=True
    #                     )

    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
