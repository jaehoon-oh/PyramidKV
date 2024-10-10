import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", \
            "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", \
            "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

datasets = ["narrativeqa", "qasper", "multifieldqa_en"]

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

model2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

model2maxlen = {
    "meta-llama-3-8b-instruct": 7950,
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def build_chat(prompt):
    prompt = f"[INST] {prompt} [/INST]"
    return prompt

def main(model, kv_compression_method, dataset, args):
    test_data = []
    prompts = []
    inputs = []
    contexts = []
    answerss = []
    lengths = []
    datasets = []
    languages = []
    all_classess = []
    _ids = []
    
    model_path = args.model_path.lower()
    model_name = model_path.split("/")[-1]
    model_max_len = model2maxlen[model_name]    
    output_max_len = dataset2maxlen[dataset]
    
    print("Loading data...")
    data_file = f"data/LongBench/{dataset}.jsonl"
    input_max_len = 0
    with open(data_file) as fp:
        for line in fp:
            example = json.loads(line) # example includes "input", "context", "answers", "length", "dataset", "language", "all_classes", "_id"
            length = example["length"]
            if length > input_max_len:
                input_max_len = length

            template = model2prompt[dataset]
            prompt = template.format(**example)            
            if "llama2" in model_path:
                prompt = build_chat(prompt)
            example["prompt"] = prompt
            test_data.append(example)
    
    print(f"Max Length in inputs is {input_max_len}")

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        if args.sample_method == "random":
            test_data = random.sample(test_data, args.max_num_examples)
        elif args.sample_method == "topk":
            test_data = test_data[:args.max_num_examples]
    
    for example in test_data:
        prompts.append(example["prompt"])
        inputs.append(example["input"])
        contexts.append(example["context"])
        answerss.append(example["answers"])
        lengths.append(example["length"])
        datasets.append(example["dataset"])
        languages.append(example["language"])
        all_classess.append(example["all_classes"])
        _ids.append(example["_id"])

    os.makedirs(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", dataset), exist_ok=True)
    fout = open(os.path.join(args.save_dir, f"{model_name}_{args.max_capacity_prompts}", dataset, f"{kv_compression_method}.json"), "w")
     
    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        batch_prompts = prompts[i:i+args.eval_batch_size]
        batch_inputs = inputs[i:i+args.eval_batch_size]
        batch_contexts = contexts[i:i+args.eval_batch_size]
        batch_answerss = answerss[i:i+args.eval_batch_size]
        batch_lengths = lengths[i:i+args.eval_batch_size]
        batch_datasets = datasets[i:i+args.eval_batch_size]
        batch_languages = languages[i:i+args.eval_batch_size]
        batch_all_classess = all_classess[i:i+args.eval_batch_size]
        batch__ids = _ids[i:i+args.eval_batch_size]
        
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if len(batch_input_ids[0]) > model_max_len:
            half = int(model_max_len/2)
            prompt = tokenizer.decode(batch_input_ids[0][:half], skip_special_tokens=True) + \
                     tokenizer.decode(batch_input_ids[0][-half:], skip_special_tokens=True) 
            
            tokenized_prompts = tokenizer(prompt, padding="longest", return_tensors="pt", add_special_tokens=True).to('cuda')
            batch_input_ids = tokenized_prompts.input_ids
            attention_mask = tokenized_prompts.attention_mask
        
        if kv_compression_method != "fullkv":
            max_capacity_prompts = args.max_capacity_prompts
            if ("snapkv" in kv_compression_method) or \
               ("pyramidkv" in kv_compression_method) or \
               ("h2o" in kv_compression_method):
                window_sizes = 16
            elif kv_compression_method in ["streamingllm"]:
                window_sizes = max_capacity_prompts - 4

            kernel_sizes = 5
            pooling = "maxpool"

            layers = len(model.model.layers)
            # check if window_sizes is a list
            if not isinstance(window_sizes, list):
                window_sizes = [window_sizes] * layers
            if not isinstance(max_capacity_prompts, list):
                max_capacity_prompts = [max_capacity_prompts] * layers
            if not isinstance(kernel_sizes, list):
                kernel_sizes = [kernel_sizes] * layers
            for i in range(layers):
                model.model.layers[i].self_attn.config.window_size = window_sizes[i]
                model.model.layers[i].self_attn.config.max_capacity_prompt = max_capacity_prompts[i]
                model.model.layers[i].self_attn.config.kernel_size = kernel_sizes[i]
                model.model.layers[i].self_attn.config.pooling = pooling

        context_length = batch_input_ids.shape[-1]
                
        output = model.generate(
            **tokenized_prompts,
            do_sample=False,
            max_new_tokens=output_max_len,
            min_length=context_length+1,
            output_attentions = args.output_attentions,
            num_beams=1,
            temperature=1.0,
            eos_token_id=[tokenizer.eos_token_id]
        )

        batch_outputs = tokenizer.batch_decode([output[0][context_length:]], skip_special_tokens=True)        
        # print(f"debbug batch_outputs {batch_outputs}")
        torch.cuda.empty_cache()

        for j in range(args.eval_batch_size):
            example = {}
            example["prompt"] = batch_prompts[j]
            example["input"] = batch_inputs[j]
            example["context"] = batch_contexts[j]
            example["answers"] = batch_answerss[j]
            example["length"] = batch_lengths[j]
            example["dataset"] = batch_datasets[j]
            example["language"] = batch_languages[j]
            example["all_classes"] = batch_all_classess[j]
            example["_id"] = batch__ids[j]
            example["pred"] = batch_outputs[j]
            fout.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="")
    
    # Model and Tokenizer
    parser.add_argument("--model_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--use_fast_tokenizer", type=bool, default=True, help="")

    # Generation
    parser.add_argument("--output_attentions", type=bool, default=False, help="")

    # Evaluation
    parser.add_argument("--save_dir", type=str, default="result_longbench")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    
    # KV cache compression
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--max_capacity_prompts", type=int, default=512, help="")

    # For debugging (e.g., --max_num_examples 10, otherwise 200)
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--sample_method", type=str, default="topk", choices=["random", "topk"], help="how to sample the examples.")

    args = parser.parse_args()
    
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(
        args.model_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=args.use_fast_tokenizer,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    from pyramidkv.monkeypatch import replace_llama, replace_mistral
    kv_compression_method = args.method.lower()
    replace_llama(kv_compression_method)
    replace_mistral(kv_compression_method)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=config.torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation
    )
    model.config.kv_compression_method = kv_compression_method
    model.eval()

    for idx, dataset in enumerate(datasets):
        print(f"Working on max_capacity_prompts {args.max_capacity_prompts} dataset {dataset} - {idx}/{len(datasets)}")
        main(model, kv_compression_method, dataset, args)
