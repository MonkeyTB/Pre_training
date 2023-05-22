import sys
import os
import argparse
from loguru import logger
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from chatglm.chatglm_model import ChatGlmModel


def preprocess_batch_for_hf_dataset(example, tokenizer, args):
    input_text, target_text = example["question"], example["answer"]
    instruction = '根据下面的职位名称和职位内容，生成对应的搜索条件，并输出json格式：'
    prompt = f"问：{instruction}\n{input_text}\n答："
    prompt_ids = tokenizer.encode(prompt, max_length=args.max_seq_length, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, max_length=args.max_length, add_special_tokens=False)
    input_ids = prompt_ids + [tokenizer.bos_token_id, tokenizer.gmask_token_id] + target_ids
    input_ids = input_ids[:(args.max_seq_length + args.max_length)] + [tokenizer.eos_token_id]

    example['input_ids'] = input_ids
    return example


class AdgDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = load_from_disk(data)
        else:
            dataset = load_dataset(data)
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset["train"]
        dataset = dataset.map(
            lambda x: preprocess_batch_for_hf_dataset(x, tokenizer, args),
            batched=False, remove_columns=dataset.column_names
        )
        dataset.set_format(type="np", columns=["input_ids"])

        self.examples = dataset["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default="data/train_4.json", type=str, help='Datasets name')
    parser.add_argument('--model_type', default='chatglm', type=str, help='Transformers model type')
    parser.add_argument('--model_name', default='../../THUDM/chatglm-6b', type=str, help='Transformers model or path')
    parser.add_argument('--do_train',   default=True, action='store_true', help='Whether to run training.')
    parser.add_argument('--output_dir', default='./result_v3/', type=str, help='Model output directory')
    parser.add_argument('--max_seq_length', default=1500, type=int, help='Input max sequence length')
    parser.add_argument('--max_length', default=100, type=int, help='Output max sequence length')
    parser.add_argument('--num_epochs', default=1, type=float, help='Number of training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    args = parser.parse_args()
    logger.info(args)
    model = None
    # fine-tune chatGLM model
    if args.do_train:
        logger.info('Loading data...')
        model_args = {
            "dataset_class": AdgDataset,
            'use_lora': True,
            "overwrite_output_dir": True,
            "max_seq_length": args.max_seq_length,
            "max_length": args.max_length,
            "per_device_train_batch_size": args.batch_size,
            "num_train_epochs": args.num_epochs,
            "output_dir": args.output_dir,
        }
        model = ChatGlmModel(args.model_type, args.model_name, args=model_args, cuda_device = '0')
        model.train_model(args.train_file)


if __name__ == '__main__':
    main()