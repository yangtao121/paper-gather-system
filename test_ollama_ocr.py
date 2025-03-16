from pix2text import Pix2Text

img_fp = '/home/ai-server/dev/paper-gather-system/src/papers/nomad/nomad.pdf'
p2t = Pix2Text.from_config()
doc = p2t.recognize_pdf(img_fp, page_numbers=[0, 1])
doc.to_markdown('output-md')  # 导出的 Markdown 信息保存在 output-md 目录中