from ollama_ocr import OCRProcessor

ocr = OCRProcessor(
    base_url="http://192.168.5.72:11434/api/generate",
    model_name="minicpm-v"
)

result = ocr.process_image(
    image_path="/Users/yangtao/Documents/code.nosync/paper-gather-system/src/papers/Exploring_the_Generalizability_of_Geomagnetic_Navigation__A_Deep_Reinforcement_Learning_approach_with_Policy_Distillation/segments/page4/007_List.png",
    # custom_prompt="You are tasked with analyzing the first page of a document. Based on the content provided, classify the document into an appropriate category, such as a research paper, thesis, dissertation, or any other formal academic document. Focus on identifying key structural or contextual clues that indicate the document’s type."
)

print("result:")
print(result)


# curl - -location 'http://localhost:11434/api/generate' \
#     - -header 'Content-Type: text/plain' \
#     - -data '{"model": "llama3.2-vision:11b", "prompt": "create a codeigniter form", "stream": false}
# '
