import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline
import os

# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf" #"microsoft/Phi-3-mini-4k-instruct"
# MODEL_NAME = "EleutherAI/gpt-j-6B"

if os.environ.get('LOCAL_RANK') is not None:
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
else:
    device_map = None

model_path = '/home/haokunch/home/haokunch/data/locus/llm_weights/zhilif/TOFU/ft_epoch5_lr1e-05_llama3.1-8b-inst_full_wd0.01/checkpoint-312/grad_ascent_1e-05_forget05_5/checkpoint-15'
config = AutoConfig.from_pretrained(MODEL_NAME, 
                                    trust_remote_code = True, 
                                    device_map=device_map)
model, tok = (
    AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

input_strings = ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n  \
                 You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|> Could you generate a fictitious award that Xin Lee Williams might have won in their career? <|eot_id|><|start_header_id|>assistant<|end_header_id|>']
left_pad_tokenizer = tok
left_pad_tokenizer.padding_side = 'left'
left_pad_tokenizer.padding_size = 'longest'
left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
#now generate
out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=100, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)

print(strs)


