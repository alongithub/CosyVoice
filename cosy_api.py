import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import random
import torch
import torchaudio
import logging
import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# from tools.asr import funasr_asr 

logging.getLogger('matplotlib').setLevel(logging.WARNING)


cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

def generate_seed():
    seed = random.randint(1, 100000000)
    return seed
    

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech
    
prompt_sr, target_sr = 16000, 22050

def generate_audio(
    tts_text: str, 
    mode: int = 1, # 模式： 1 中文[同语言克隆] 2 中日英混合[跨语言克隆]
    prompt_text: str = "", # 参考文本：传入或自动推
    prompt_wav: str = "",  # 参考音频
    seed: int = 0, # 随机种子
    output_path: str = "" # 输出路径
):
    
    if output_path == "":
        # Error 输出路径不能为空
        return Exception("输出路径不能为空")
    
    if tts_text == "":
        return Exception("推理文本不能为空")
    
    if prompt_wav == "":
        # ERROR 参考音频不能为空
        return Exception("参考音频不能为空")
    
    if prompt_text == "":
        # TODO 自动推参考文本
        # ERROR 参考文本不能为空
        return Exception("参考文本不能为空")
    
    if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
        # TODO 自动转换采样率
        return Exception("采样率低于 16000，请先转换采样率")
   
    if mode == 1:
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode == 2:
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)

    torchaudio.save(
        output_path,
        output['tts_speech'],
        sample_rate=target_sr
    )
    return output_path


def read_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content 
    
def sorted_file_name(dir: str, ext: str, index = 0):
    try:
        all_files = os.listdir(dir)
        files = [f for f in all_files if f.endswith(ext)]
        
        return dir + "/" + files[index]
    except Exception as e:
        print(f"error {e}")
        return ""

app = FastAPI()

class TTSInferRequest(BaseModel):
    text: str
    model_name: str
    output_path: str
    mode: int = 1


@app.post("/infer")
async def cosy_infer(
    body: TTSInferRequest = Body(...),
):  
    tts_text = body.text
    model_name = body.model_name
    output_path = body.output_path
    mode = body.mode
    
    # TODO 检查是否存在其他语言
    
    # if prompt_text is None or prompt_text == "":
    #     prompt_text = funasr_asr.only_asr(prompt_wav)
    #     logging.info("funasr asr: " + prompt_text)
    # if model_name is None or model_name == '':
    #     JSONResponse({ "message": "model name"}, status_code=400)
        
        
    seed = generate_seed()
    model_dir = os.path.join("mercury_workspace", model_name)
    if not os.path.exists(model_dir):
        return JSONResponse({ "message": "audio model v2 not exists"}, status_code=400)
    
    refer_wav_path = sorted_file_name(model_dir, '.wav') # TODO reference 的参考音频进行评估打分，仅保留只有中文或英文的参考，参考语言根据文本推断
    refer_lab_path = sorted_file_name(model_dir, '.lab') # TODO 根据参考音频的名称选取
    prompt_text = read_file(refer_lab_path)
    
    return generate_audio(tts_text, mode, prompt_text, refer_wav_path, seed, output_path)
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3335)