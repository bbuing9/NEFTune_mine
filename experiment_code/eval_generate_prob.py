"""This script generates the evaluation responses that can e used by eval_scoring.py"""
import argparse
from functools import partial
import json

from datasets import Dataset
import datasets
import transformers
from transformers.models.llama.configuration_llama import LlamaConfig
import torch
from torch.utils.data import DataLoader

from io_utils import load_jsonlines
from utils import load_fsdp_ckpt_with_accelerate, add_padding_token
from conversation import get_conv_template

from dataset import make_supervised_data_module
from train import get_dataloader_and_sampler
from tqdm import tqdm
import numpy as np 

def apply_conv_template(example, template_type):
    # preprocess instructions into prompted inputs
    conv = get_conv_template(template_type)
    conv.append_message(conv.roles[0], example['instruction'])
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    example.update({
        "prompt": prompt
    })

    return example

def generate_responses_batched(example, model, tokenizer, kwargs):
    prompt = example['prompt']
    print(prompt)
    encoding = tokenizer(prompt, 
                          return_tensors="pt",
                          padding="longest",
                          max_length=tokenizer.model_max_length,
                          truncation=True,
                      )
    encoding = encoding.to(model.device)
    with torch.no_grad():
        model_output = model.generate(**encoding, **kwargs)
        input_len = encoding.input_ids.shape[-1]
        model_output = model_output[:, input_len:].cpu()
        decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    del example['prompt']
    example.update({"output": decoded_output}) 
    example.update({"metadata": [kwargs] * len(decoded_output)})

    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama/7B_sharded", type=str)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--model_config_path", default="llama/7B_hf", type=str)
    parser.add_argument("--template_type", default="alpaca", type=str)
    parser.add_argument("--file_path", default="datasets/self-instruct-val(processed).jsonl", type=str)
    parser.add_argument("--save_file_name", default="outputs/answers/self-instruct_llama7B.jsonl", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--debug", action='store_true', help="This reduce the number of generation examples to 4, so that we can debug faster.")
    args = parser.parse_args()

    model_config = transformers.AutoConfig.from_pretrained(args.model_config_path)
    if isinstance(model_config, LlamaConfig):
        model_config.vocab_size += 1 # hardcode the vocab size for llama... 

    model = load_fsdp_ckpt_with_accelerate(args.model, model_config, hf_dummy_path=args.model_config_path, wrapped_class="LlamaDecoderLayer" if 'llama' in args.model else "OPTDecoderLayer")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_config_path,
            model_max_length=2048,
            padding_side="left",
            use_fast=False,
        )
    add_padding_token(tokenizer)
    
    ## set the models to eval mode
    model = model.eval()
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path='datasets/alpaca-train.jsonl', data_fraction=1.0, seed=42)
    train_dataset = data_module['train_dataset']
    data_collator = data_module['data_collator']
    epoch_iterator = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=data_collator, drop_last=False, num_workers=0, pin_memory=True)

    res = torch.BFloat16Tensor(12002, 1536, 100)
    res2 = torch.LongTensor(12002, 1536, 100)

    for i, data in tqdm(enumerate(epoch_iterator)):
        if i < 10000:
            continue
        #if i == 10000:
        #    break
        with torch.no_grad():
            out = model(**data)
            logits = out.logits
            shift_targets = logits[..., 1:, :].contiguous()
            sorted_targets, sorted_indices = torch.sort(shift_targets, descending=True)
            sorted_targets = sorted_targets[:, :, :100]
            res[4*(i-10000):4*((i-10000)+1), :sorted_targets.shape[1], :] = sorted_targets.cpu()
            res2[4*(i-10000):4*((i-10000)+1), :sorted_targets.shape[1], :] = sorted_indices[:, :, :100].cpu()

    torch.save(res, './checkpoints/full_prob_50000.pt')
    torch.save(res2, './checkpoints/full_prob_50000_indices.pt')

