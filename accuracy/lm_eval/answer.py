import argparse
import json, tqdm
import torch
import copy
import os, sys
import math

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'ProgramName',
                        description = 'What the program does',
                        epilog = 'Text at the bottom of help')

    parser.add_argument('--input-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-type', type=str, default='opt')

    # Quant.
    parser.add_argument('--enable_quant', action='store_true')
    parser.add_argument("--qbits", type=int, default=8)

    # H2O
    parser.add_argument('--enable_small_cache', action='store_true')
    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    # InfiniGen
    parser.add_argument('--ours', action='store_true')
    parser.add_argument("--partial_weight_ratio", type=float, default=0.1)
    parser.add_argument("--partial_weight_path", type=str)
    parser.add_argument("--skewing_matrix_path", type=str)
    parser.add_argument("--alpha",type=float, default=5)
    parser.add_argument("--capacity",type=float, default=1.0)
    parser.add_argument("--budget",type=float, default=0.2)
    args = parser.parse_args()
    
    if args.ours:
        set_symlink(args.model_type, f"modeling_{args.model_type}_ours.py")
    else:
        set_symlink(args.model_type, f"modeling_{args.model_type}_orig.py")

    model_name = args.model_name

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
    if args.model_path is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

    if args.enable_quant:
        if args.model_type == "opt":
            for i, layer in enumerate(model.model.decoder.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits
        if args.model_type == "llama":
            for i, layer in enumerate(model.model.layers):
                if i>=2:
                    layer.self_attn.enable_quant = True
                    layer.self_attn.qbits = args.qbits

    elif args.enable_small_cache:
        from utils_lm_eval.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
        from utils_lm_eval.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
        from utils_lm_eval.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask
        ENABLE_Heavy_Hitter_FUNCTIONS = {
            "llama": convert_kvcache_llama_heavy_recent,
            "opt": convert_kvcache_opt_heavy_recent,
            "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
        }
        print('Enable Small Cache Size')
        config.heavy_ratio = args.heavy_ratio
        config.recent_ratio = args.recent_ratio
        base_path = os.path.basename(args.model_name)
        if not os.path.exists(f"../h2o_model/{base_path}.pt"):
            os.system("mkdir ../h2o_model")
            checkpoint = copy.deepcopy(model.state_dict())
            torch.save(checkpoint, f"../h2o_model/{base_path}.pt")
        model = ENABLE_Heavy_Hitter_FUNCTIONS[args.model_type](model, config)
        model.load_state_dict(torch.load(f"../h2o_model/{base_path}.pt"))
        model = model.to(torch.float16)
    
    elif args.ours:
        if args.model_type == "opt":
            for layer in range(len(model.model.decoder.layers)):
                model.model.decoder.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.decoder.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.decoder.layers[layer].self_attn.alpha = args.alpha
                model.model.decoder.layers[layer].self_attn.capacity = args.capacity
                model.model.decoder.layers[layer].self_attn.budget = args.budget
        if args.model_type == "llama":
            if args.skewing_matrix_path is not None:
                A = torch.load(args.skewing_matrix_path)
            for layer in range(len(model.model.layers)):
                model.model.layers[layer].self_attn.partial_weight_ratio = args.partial_weight_ratio
                model.model.layers[layer].self_attn.partial_weight_q = torch.load(args.partial_weight_path + "/partial_weight_q_" + str(layer) + ".pt")
                model.model.layers[layer].self_attn.alpha = args.alpha
                model.model.layers[layer].self_attn.capacity = args.capacity
                model.model.layers[layer].self_attn.budget = args.budget
                if args.skewing_matrix_path is not None:
                    model.model.layers[layer].self_attn.skewing_matrix = A[layer]

    model.half().eval().cuda()

    # with torch.no_grad():
    prompt = "Beijing is the capital of"
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
    max_length = 128

    generate_ids = model.generate(input_ids, max_new_tokens=max_length)

    result = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print("Prompt:", prompt)
    print("\n")
    print("Generated text:")
    print(result)
    
    if args.ours:
        if args.model_type == "opt":
            for layer in model.model.decoder.layers:
                layer.self_attn.previous_hidden_states = None
        if args.model_type == "llama":
            for layer in model.model.layers:
                layer.self_attn.previous_hidden_states = None
