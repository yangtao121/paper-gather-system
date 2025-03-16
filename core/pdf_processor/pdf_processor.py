import fitz
from loguru import logger
from tqdm import tqdm
import os
from core.image_processor import PDFImageProcessor


class PDFProcessor:
    def __init__(self, pdf_workspace: str):
        """
        保存文件系统：
        - doc title dir
          - .pdf
          - images
            - page1.png
            - page2.png
          - segments
            - page1
              - title.png
              - abstract.png
              - introduction.png
              - ...
            - page2
              - title.png
              - abstract.png
              - introduction.png
              - ...
        """

        # 文件系统
        self.pdf_workspace = pdf_workspace
        self.images_workspace = os.path.join(self.pdf_workspace, "images")
        self.segments_workspace = os.path.join(self.pdf_workspace, "segments")

        # logger.info(f'pdf workspace {self.pdf_workspace}')

        # 确保文件夹存在
        if not os.path.exists(self.images_workspace):
            logger.info('Create images workspace!')
            os.makedirs(self.images_workspace)

        if not os.path.exists(self.segments_workspace):
            logger.info('Create segments workspace!')
            os.makedirs(self.segments_workspace)

        # print(self.pdf_path)

        # 查找pdf文件
        self.pdf_path = None

        for root, dirs, files in os.walk(self.pdf_workspace):
            for file in files:
                if file.endswith(".pdf"):
                    self.pdf_path = os.path.join(root, file)
                    break
            if self.pdf_path:
                break
        if not self.pdf_path:
            logger.error("No PDF file found.")

        # 打开PDF文件
        try:
            self.pdf = fitz.open(self.pdf_path)
            logger.info(f"Successfully opened PDF: {self.pdf_path}")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise e

        self.image_paths = []
        self.pdf_image_processors = []

    def convert_to_image(self, 
                         process_image: bool = True,
                         image_enhance: bool = False,
                         dpi=300,
                         fix_size: bool = False,
                         fix_length: int = 672):
        """
        将PDF转换为图片。


        """
        # 1. 创建每一页的图片
        logger.info("Converting PDF to images...")

        # 计算缩放矩阵，基于DPI
        zoom = dpi / 72  # 72是PDF的默认DPI
        matrix = fitz.Matrix(zoom, zoom)

        with tqdm(total=len(self.pdf)) as pbar:
            for page in self.pdf:
                # 使用高质量设置获取像素图
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = os.path.join(self.images_workspace,
                                          f"page{page.number + 1}.png")
                try:
                    pix.save(image_path)
                except Exception as e:
                    logger.error(f"Failed to save image: {e}")
                    raise e
                self.image_paths.append(image_path)
                # 在segments_workspace中创建一个文件夹，文件夹名称为page.number + 1
                segment_path = os.path.join(self.segments_workspace,
                                            f"page{page.number + 1}")
                os.makedirs(segment_path, exist_ok=True)
                
                if process_image:
                    image_processor = PDFImageProcessor(
                        image_path=image_path,
                        segments_workspace=segment_path,
                        image_type="research paper",
                        image_enhance=image_enhance,
                        fix_size=fix_size,
                        fix_length=fix_length,
                    )
                    # 将图片进行分割
                    image_processor.pdf_image_process()
                    self.pdf_image_processors.append(image_processor)
                pbar.update(1)  # 更新进度条
                
            

        return self.image_paths


if __name__ == "__main__":
    pdf_processor = PDFProcessor(
        "/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation",
    )
    pdf_processor.convert_to_image(process_image=True)
