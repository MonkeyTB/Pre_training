import sys
import os
import argparse
from loguru import logger
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from chatglm.chatglm_model import ChatGlmModel
import platform
import signal

CUDA_VISIBLE_DEVICES = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="data/train.json", type=str, help='Datasets name')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='../../THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_predict', default=True, action='store_true', help='Whether to run predict.')
    parser.add_argument('--output_dir', default='./result_v2/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=1500, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=100, type=int, help='Output max sequence length')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    if args.do_predict:
        if model is None:
            model = ChatGlmModel(
                args.model_type, args.model_name,
                args={'use_lora': True, 'output_dir': args.output_dir, "max_length": args.max_length, },
                cuda_device='1'
            )
        while True:
            history = []
            query = input("\ninput:")
            for response, history in model.stream_predict(query, history=[]):
                print(response)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_file', default="data/train.json", type=str, help='Datasets name')
#     parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
#     parser.add_argument('--model_name', default='../../THUDM/chatglm-6b', type=str, help='Transformers model or path')
#     parser.add_argument('--do_predict', default=True, action='store_true', help='Whether to run predict.')
#     parser.add_argument('--output_dir', default='./result_v2/', type=str, help='Model output directory')
#     parser.add_argument('--max_seq_length', default=1500, type=int, help='Input max sequence length')
#     parser.add_argument('--max_length', default=100, type=int, help='Output max sequence length')
#     parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
#     args = parser.parse_args()
#     logger.info(args)
#     model = None
#     if args.do_predict:
#         if model is None:
#             model = ChatGlmModel(
#                 args.model_type, args.model_name,
#                 args={'use_lora': True, 'output_dir': args.output_dir, "max_length": args.max_length,},
#                 cuda_device = '1'
#             )
#         history = []
#         global stop_stream
#         print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
#         while True:
#             query = input("\n用户：")
#             if query.strip() == "stop":
#                 break
#             if query.strip() == "clear":
#                 history = []
#                 os.system(clear_command)
#                 print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
#                 continue
#             count = 0
#             for response, history in model.stream_chat(query, history=history, max_length=600, no_repeat_ngram_size=8, repetition_penalty=1.2, typical_p=0.2 ):
#                 if stop_stream:
#                     stop_stream = False
#                     break
#                 else:
#                     count += 1
#                     if count % 8 == 0:
#                         os.system(clear_command)
#                         print(build_prompt(history), flush=True)
#                         signal.signal(signal.SIGINT, signal_handler)
#             os.system(clear_command)
#             print(build_prompt(history), flush=True)


if __name__ == '__main__':
    main()
