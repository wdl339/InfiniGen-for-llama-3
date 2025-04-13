import torch
import copy
import os

def set_symlink(model_type, fname):
    model_path = "../transformers/src/transformers/models/" + model_type
    linker_path = os.path.realpath("../src/" + fname)
    if not os.path.exists(linker_path):
        print(f"No file exists at {linker_path}")
        exit(0)
    if not os.path.exists(model_path):
        print(f"No file exists at {model_path}")
        exit(0)
    curr_dir = os.getcwd()
    os.chdir(model_path)
    if os.path.exists(f'modeling_{model_type}.py'):
        cmd = f"rm modeling_{model_type}.py"
        os.system(cmd)
    cmd = f"ln -s {linker_path} modeling_{model_type}.py"
    os.system(cmd)
    os.chdir(curr_dir)

# args_model_name = "/mnt/llama-model/llama-2-7b"
# args_model_path = "/mnt/llama-model/llama-2-7b"
# args_model_name = "/mnt/llama-model/llama-3.1-8b-instruct"
# args_model_path = "/mnt/llama-model/llama-3.1-8b-instruct"
args_model_name = "/mnt/llama-model/llama-3.2-3b-instruct"
args_model_path = "/mnt/llama-model/llama-3.2-3b-instruct"
args_model_type = "llama"
args_enable_quant = False
args_qbits = 8
args_enable_small_cache = False
args_heavy_ratio = 0.1
args_recent_ratio = 0.1
args_ours = True
# args_ours = False
args_partial_weight_ratio = 0.2
# args_partial_weight_path = "../setup/weights/llama-2-7b_0.2"
# args_skewing_matrix_path = "../setup/skewing_matrix/llama-2-7b.pt"
# args_partial_weight_path = "../setup/weights/llama-3.1-8b-instruct_0.2"
# args_skewing_matrix_path = "../setup/skewing_matrix/llama-3.1-8b-instruct.pt"
args_partial_weight_path = "../setup/weights/llama-3.2-3b-instruct_0.2"
args_skewing_matrix_path = "../setup/skewing_matrix/llama-3.2-3b-instruct.pt"
args_alpha = 7
args_capacity = 1
args_budget = 0.6

if args_ours:
    set_symlink(args_model_type, f"modeling_{args_model_type}_ours.py")
else:
    set_symlink(args_model_type, f"modeling_{args_model_type}_orig.py")

model_name = args_model_name

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
if args_model_path is None:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
else:
    model = AutoModelForCausalLM.from_pretrained(args_model_path)

if args_enable_quant:
    if args_model_type == "opt":
        for i, layer in enumerate(model.model.decoder.layers):
            if i>=2:
                layer.self_attn.enable_quant = True
                layer.self_attn.qbits = args_qbits
    if args_model_type == "llama":
        for i, layer in enumerate(model.model.layers):
            if i>=2:
                layer.self_attn.enable_quant = True
                layer.self_attn.qbits = args_qbits

elif args_enable_small_cache:
    from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
    from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
    from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
    ENABLE_Heavy_Hitter_FUNCTIONS = {
        "llama": convert_kvcache_llama_heavy_recent,
        "opt": convert_kvcache_opt_heavy_recent,
        "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
    }
    print('Enable Small Cache Size')
    config.heavy_ratio = args_heavy_ratio
    config.recent_ratio = args_recent_ratio
    base_path = os.path.basename(args_model_name)
    if not os.path.exists(f"../h2o_model/{base_path}.pt"):
        os.system("mkdir ../h2o_model")
        checkpoint = copy.deepcopy(model.state_dict())
        torch.save(checkpoint, f"../h2o_model/{base_path}.pt")
    model = ENABLE_Heavy_Hitter_FUNCTIONS[args_model_type](model, config)
    model.load_state_dict(torch.load(f"../h2o_model/{base_path}.pt"))
    model = model.to(torch.float16)

elif args_ours:
    if args_model_type == "opt":
        for layer in range(len(model.model.decoder.layers)):
            model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args_partial_weight_ratio
            model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args_partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
            model.model.decoder.layers[layer].self_attn.alpha = args_alpha
            model.model.decoder.layers[layer].self_attn.capacity = args_capacity
            model.model.decoder.layers[layer].self_attn.budget = args_budget
    if args_model_type == "llama":
        if args_skewing_matrix_path is not None:
            A = torch.load(args_skewing_matrix_path)
        for layer in range(len(model.model.layers)):
            model.model.layers[layer].self_attn.partial_weight_ratio = args_partial_weight_ratio
            model.model.layers[layer].self_attn.partial_weight_q = torch.load(args_partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
            model.model.layers[layer].self_attn.alpha = args_alpha
            model.model.layers[layer].self_attn.capacity = args_capacity
            model.model.layers[layer].self_attn.budget = args_budget
            if args_skewing_matrix_path is not None:
                model.model.layers[layer].self_attn.skewing_matrix = A[layer]

def generate_completion(chat, max_tokens: int, temperature: float) -> str:
    model.half().eval().cuda()

    with torch.no_grad():
        message = tokenizer.apply_chat_template(chat, tokenize=False)
        print(f"length: {len(message)}")
        input_ids = tokenizer(message, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        
        generate_ids = model.generate(
            input_ids, 
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.2
        )
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        if args_ours:
            if args_model_type == "opt":
                for layer in model.model.decoder.layers:
                    layer.self_attn.previous_hidden_states = None
            if args_model_type == "llama":
                for layer in model.model.layers:
                    layer.self_attn.previous_hidden_states = None

        # result - message
        return result.replace(message,"")\
                    .replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")\
                    .replace("<|eot_id|>","")

# def generate_completion(chat, max_tokens: int, temperature: float) -> str:
#     model.half().eval().cuda()

#     with torch.inference_mode():
#         message = tokenizer.apply_chat_template(chat, tokenize=False)
#         input_ids = tokenizer.encode(message, return_tensors="pt").to(model.device)
        
#         generate_ids = model.generate(
#             input_ids, 
#             max_new_tokens=max_tokens,
#             temperature=temperature,
#             repetition_penalty=1.2
#         )
#         output = generate_ids.cpu()
#         result = tokenizer.decode(output[0], skip_special_tokens=False)

#         if args_ours:
#             if args_model_type == "opt":
#                 for layer in model.model.decoder.layers:
#                     layer.self_attn.previous_hidden_states = None
#             if args_model_type == "llama":
#                 for layer in model.model.layers:
#                     layer.self_attn.previous_hidden_states = None

#         # result - message
#         return result.replace(message,"")\
#                     .replace("<|start_header_id|>assistant<|end_header_id|>\n\n","")\
#                     .replace("<|eot_id|>","")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import uvicorn
import threading

app = FastAPI()
lock = threading.Lock()

class CompletionRequest(BaseModel):
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stop: Optional[str] = None
    temperature: Optional[float] = None
    prompt: Optional[str] = ""

    # "messages": [
    #     {
    #         "content": "...",
    #         "role": "user"
    #     }
    # ],

class CompletionResponseChoice(BaseModel):
    message: Dict[str, str]

class CompletionResponse(BaseModel):
    choices: List[CompletionResponseChoice]

@app.post("/v1/chat/completions")
async def create_completion(request: CompletionRequest):
    try:
        lock.acquire()
        completion = generate_completion(
            request.messages,
            request.max_tokens,
            request.temperature
        )
        lock.release()

        return CompletionResponse(
            choices=[
                CompletionResponseChoice(
                    message={"content": completion}
                )
            ]
        )
    
    except Exception as e:
        lock.release()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)