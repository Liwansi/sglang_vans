import json
from transformers import AutoTokenizer
import os


def create_data(input_len, batch_size, model_path, save_path):
    fix_input_len = 204800#25600
    dynamic_input_len = 51200#6400
    input_len = 256000#32000
    batch_size = 30
    model_path = '/data/ascend-ci-share-pkking-sglang/modelscope/hub/models/zcgy26/Qwen3-235B-A22B-Instruct-2507-w8a8' 
    save_path = '/home/lws/sglang/aisbench_auto_tools_prefix/dataset' 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if os.path.exists(f'GSM8K-in{input_len}-bs{batch_size}.jsonl'):
        print("dataset already exists...")
        exit(0)

    if not os.path.exists('./GSM8K.jsonl'):
        print("gsm8k dataset not exists...")
        exit(0)

    dataset = []
    with open('./GSM8K.jsonl', 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data['question'])

    # repeat input_len
    dataset_2k = []
    fix_question = dataset[0]
    fix_tokens = tokenizer.tokenize(fix_question)
    if len(fix_tokens) < fix_input_len:
        repeat_times = (fix_input_len // len(fix_tokens)) + 1
        fix_tokens = (fix_tokens * repeat_times)[:fix_input_len]
    else:
        fix_tokens = fix_tokens[:fix_input_len]


    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        if len(words) == 0:
            continue
        len_num = len(words) // dynamic_input_len
        #print(f"{len_num=} {len(words)=}")
        if len_num == 0:
            multiplier = (dynamic_input_len // len(words)) + 1
            repeated_len = words * multiplier
            words = repeated_len[:dynamic_input_len]
            print(f"{len(words)=} {len(fix_tokens)=}")
            full_words = fix_tokens + words
            decoded_text = tokenizer.convert_tokens_to_string(full_words)
            dataset_2k.append(decoded_text)

    # repeat to batch_size
    batch_num = len(dataset_2k) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_2k)) + 1
        repeated_batch = dataset_2k * multiplier
        dataset_2k = repeated_batch[:batch_size]
    else:
        dataset_2k = dataset_2k[:batch_size]

    json_str = json.dumps(dataset_2k, ensure_ascii=False, indent=4)
    base_name = os.path.basename(os.path.normpath(model_path))
    with open(os.path.join(save_path, f'GSM8K-in{input_len}-bs{batch_size}-{base_name}.jsonl'), 'w', encoding='utf-8') as f:
        for i in range(len(dataset_2k)):
            f.write(json.dumps({"question": dataset_2k[i], "answer": "none"}, ensure_ascii=False))
            f.write("\n")

create_data(1, 1, '', '')
