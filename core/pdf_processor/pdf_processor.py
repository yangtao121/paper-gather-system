import fitz
from loguru import logger
from tqdm import tqdm
import os
from core.image_processor import PDFImageProcessor
import numpy as np

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
            logger.info("Create images workspace!")
            os.makedirs(self.images_workspace)

        if not os.path.exists(self.segments_workspace):
            logger.info("Create segments workspace!")
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

        # # 判断是否为双栏排版
        
        # for page_number in range(len(self.pdf)):
        #     self._is_two_column = self.is_two_column(
        #         pdf_path=self.pdf_path,
        #         page_number=page_number,
        #         gap_threshold=0.15,
        #         mid_region=0.25,
        #         min_text_blocks=5,
        #         debug=True,
        #     )
        #     logger.info(f"The PDF is two column: {self._is_two_column}")

        self.image_paths = []
        self.pdf_image_processors = []

    def convert_to_image(
        self,
        process_image: bool = True,
        image_enhance: bool = False,
        dpi=300,
        fix_size: bool = False,
        fix_length: int = 672,
    ):
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
                image_path = os.path.join(
                    self.images_workspace, f"page{page.number + 1}.png"
                )
                try:
                    pix.save(image_path)
                except Exception as e:
                    logger.error(f"Failed to save image: {e}")
                    raise e
                self.image_paths.append(image_path)
                # 在segments_workspace中创建一个文件夹，文件夹名称为page.number + 1
                segment_path = os.path.join(
                    self.segments_workspace, f"page{page.number + 1}"
                )
                os.makedirs(segment_path, exist_ok=True)

                if process_image:
                    image_processor = PDFImageProcessor(
                        image_path=image_path,
                        page_number=page.number+1,
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

    def is_two_column(
        self,
        page_number=0,
        gap_threshold=0.15,
        mid_region=0.25,
        min_text_blocks=5,
        debug=False,
    ):
        """
        改进版PDF双栏排版检测函数

        参数改进说明：
        - gap_threshold: 间隙需至少占页面宽度的15%（原10%）
        - mid_region: 中间区域范围缩小到页面宽度的25%（原30%）
        - min_text_blocks: 要求至少5个文本块才进行分析
        - debug: 打印调试信息
        """
        # doc = fitz.open(pdf_path)
        page = self.pdf.load_page(page_number)
        width = page.rect.width
        height = page.rect.height

        # 提取所有文本块（不再过滤类型）
        blocks = page.get_text("blocks")
        if len(blocks) < min_text_blocks:
            if debug:
                print("文本块不足，无法分析")
            return False

        # 过滤页眉页脚（只保留页面中间70%的垂直区域）
        vertical_margin = height * 0.15
        blocks = [
            b
            for b in blocks
            if vertical_margin < (b[1] + b[3]) / 2 < height - vertical_margin
        ]

        # 收集所有文本块的左右边界
        x_coords = []
        for b in blocks:
            x_coords.extend([b[0], b[2]])

        if len(x_coords) < 4:
            return False

        x_coords = np.array(sorted(x_coords))

        # 计算相邻坐标的间隙
        gaps = x_coords[1:] - x_coords[:-1]
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]
        gap_start = x_coords[max_gap_idx]
        gap_end = x_coords[max_gap_idx + 1]

        # 计算中间区域
        mid_x = width / 2
        region_start = mid_x - (mid_region * width / 2)
        region_end = mid_x + (mid_region * width / 2)

        # 判断条件
        condition1 = (gap_start > region_start) and (gap_end < region_end)
        condition2 = max_gap > (width * gap_threshold)

        if debug:
            print(f"页面宽度: {width:.1f}")
            print(f"最大间隙: {max_gap:.1f} ({max_gap/width:.1%})")
            print(f"间隙位置: {gap_start:.1f} - {gap_end:.1f}")
            print(f"中间区域: {region_start:.1f} - {region_end:.1f}")
            print(f"条件1（位置）: {condition1}")
            print(f"条件2（宽度）: {condition2}")

        return condition1 and condition2


if __name__ == "__main__":
    pdf_processor = PDFProcessor(
        "/home/ai-server/dev/paper-gather-system/src/papers/nomad",
    )
    pdf_processor.convert_to_image(process_image=True)
