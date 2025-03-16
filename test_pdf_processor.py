from core.pdf_processor import PDFProcessor


pdf_processor = PDFProcessor(
    "/home/ai-server/dev/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation",
)
pdf_processor.convert_to_image(process_image=True,image_enhance=False,fix_size=False,fix_length=672)