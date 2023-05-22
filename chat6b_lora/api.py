from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from chatglm.chatglm_model import ChatGlmModel
import argparse

DEVICE = 'cuda'
DEVICE_ID = '1'
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post('/')
async def create_item(request: Request):
    global model
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    repetition_penalty = json_post_list.get('repetition_penalty')
    no_repeat_ngram_size = json_post_list.get('no_repeat_ngram_size')
    typical_p = json_post_list.get('typical_p')
    response = model.chat(
        [prompt],
        history=history,
        max_length=max_length if max_length else 600,
        repetition_penalty=1.2,
        top_p=top_p if top_p else 0.7,
        temperature=temperature if temperature else 0.95,
        typical_p=typical_p if typical_p else 0.2,
        no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size else 8,
    )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        'response': response,
        'status': 200,
        'time': time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='../../THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--output_dir', default='./result/', type=str, help='Model output directory')
    args = parser.parse_args()

    model = ChatGlmModel(
        args.model_type, args.model_name,
        args={'use_lora': True, 'output_dir': args.output_dir},
        cuda_device=DEVICE_ID
    )
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

