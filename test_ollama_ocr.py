from ollama_ocr import OCRProcessor

ocr = OCRProcessor(
    base_url="http://192.168.5.72:11434/api/generate"
)

result = ocr.process_image(
    image_path="/Users/yangtao/Documents/code.nosync/paper-gather-system/src/papers/Exploring the Generalizability of Geomagnetic Navigation_ A Deep Reinforcement Learning approach with Policy Distillation.pdf_page0.png",
    # format_type="json",
    custom_prompt="You are tasked with analyzing the first page of a document. Based on the content provided, classify the document into an appropriate category, such as a research paper, thesis, dissertation, or any other formal academic document. Focus on identifying key structural or contextual clues that indicate the document’s type."
)


print(result)


# curl - -location 'http://localhost:11434/api/generate' \
#     - -header 'Content-Type: text/plain' \
#     - -data '{"model": "llama3.2-vision:11b", "prompt": "create a codeigniter form", "stream": false}
# '
