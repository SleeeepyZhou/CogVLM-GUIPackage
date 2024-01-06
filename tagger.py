# img2cap for cogagent-vqa

import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from transformers.utils.import_utils import is_torch_bf16_gpu_available
import argparse
from tqdm import tqdm

# 设置 argparse 参数
parser = argparse.ArgumentParser()
parser.add_argument("--from_pretrained", type=str, default="./cogagent-vqa-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--image_folder", type=str, default="./test", help='Image folder path')
query = "Describe this image in a very detailed manner."

args = parser.parse_args()

MODEL_PATH =  "./models/cogagent-vqa-hf"
TOKENIZER_PATH = args.local_tokenizer
IMAGE_FOLDER = args.image_folder
OUTPUT_FOLDER = args.image_folder

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

if is_torch_bf16_gpu_available():
    torch_type = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
       MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    torch_type = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
       MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()


# 支持的图像格式
image_extensions = ['.png', '.jpg', '.jpeg', '.webp']

# 扫描文件夹并筛选图像文件
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in image_extensions]

for image_path in tqdm(image_files, desc="Processing images"):
    image = Image.open(image_path).convert('RGB')
    history = []
    # 构建模型输入
    input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    # 设置生成参数
    gen_kwargs = {"max_length": 2048, "temperature": 1, "do_sample": False}

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("</s>")[0]
        response = response.replace("<EOI>Question: " + query + " Answer: ", "")

    # 打印响应
    # print(f"\nCog: {response}")

    # 保存响应到指定的输出目录
    output_file_name = os.path.basename(image_path)
    output_file_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(output_file_name)[0] + ".txt")
    with open(output_file_path, "w") as file:
        file.write(response)

    history.append((query, response))
