import torch
import numpy as np

from text_wrapper_craftax import *
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_llama import CustomLlamaForCausalLM
from concurrent.futures import ThreadPoolExecutor

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# rank = int(os.environ["RANK"])
# device = torch.device(f"cuda:{rank}")
# torch.cuda.set_device(device)
# torch.distributed.init_process_group("nccl", device_id=device)

# dist.init_process_group(backend="nccl")
# rank = dist.get_rank()
# device = rank % torch.cuda.device_count()

# device = torch.device("cuda:1")

emb_dict_map = {0: "mean", 1: "exp"}

num_gpus = torch.cuda.device_count()
local_dir = "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct/"

llm_pretrained_all = [
    CustomLlamaForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    .to(f"cuda:{i}")
    .eval()
    for i in range(1, num_gpus)
]
tokenizer = AutoTokenizer.from_pretrained(local_dir)

for llm_pretrained in llm_pretrained_all:
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    llm_pretrained.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    embedding_dim = llm_pretrained.get_input_embeddings().weight.shape[1]
    padding_token_id = tokenizer.convert_tokens_to_ids("<pad>")

    with torch.no_grad():
        llm_pretrained.get_input_embeddings().weight[padding_token_id] = torch.zeros(
            embedding_dim
        )

llm_pretrained_all = [torch.compile(llm_pretrained_all[i]) for i in range(num_gpus - 1)]


def gpu_inference(i, text_obs_chunk, layer, emb_type):
    with torch.no_grad():
        batch_tokens = tokenizer(
            text_obs_chunk,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        ).to(llm_pretrained_all[i].device)

        hidden_states, _ = llm_pretrained_all[i].forward_hidden_states(
            **batch_tokens,
            use_cache=True,
            output_hidden_states=True,
            target_layer=[layer],
            emb_type=emb_dict_map[emb_type],
        )

    return hidden_states[0]


def get_llm_obs(obs, layer, emb_type):
    obs = np.array(obs)
    text_obs = []
    for curr_obs in obs:
        curr_text_list = symbolic_to_text_numpy(curr_obs)
        curr_text = "\n".join(curr_text_list)
        text_obs.append(curr_text)

    text_obs_chunks = [text_obs[i :: num_gpus - 1] for i in range(num_gpus - 1)]

    embed = []
    with ThreadPoolExecutor(max_workers=num_gpus - 1) as executor:
        futures = [
            executor.submit(gpu_inference, i, text_obs_chunks[i], layer, emb_type)
            for i in range(num_gpus - 1)
        ]
        for future in futures:
            embed.append(future.result().cpu().numpy().astype(np.float32))

    numpy_embed = np.concatenate(embed, axis=0)

    torch.cuda.empty_cache()

    return numpy_embed


# local_dir = "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct/"

# # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# # llm_pretrained_raw = AutoModelForCausalLM.from_pretrained(local_dir, quantization_config=quantization_config, device_map="balanced_low_0", low_cpu_mem_usage=True)

# llm_pretrained_raw = CustomLlamaForCausalLM.from_pretrained(
#     local_dir,
#     torch_dtype=torch.float16,
#     device_map="balanced_low_0",
#     low_cpu_mem_usage=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(local_dir, device_map="balanced_low_0")

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
#     llm_pretrained_raw.resize_token_embeddings(len(tokenizer), mean_resizing=False)
#     embedding_dim = llm_pretrained_raw.get_input_embeddings().weight.shape[1]
#     padding_token_id = tokenizer.convert_tokens_to_ids("<pad>")

#     with torch.no_grad():
#         llm_pretrained_raw.get_input_embeddings().weight[
#             padding_token_id
#         ] = torch.zeros(embedding_dim)

# # llm_pretrained_raw = DDP(llm_pretrained_raw, device_ids=[device])
# # llm_pretrained = llm_pretrained_raw
# llm_pretrained = torch.compile(llm_pretrained_raw, fullgraph=True)

# def get_llm_obs(obs, embeddding_type, layer, k):
#     obs = np.array(obs)
#     text_obs = []
#     for curr_obs in obs:
#         curr_text_list = symbolic_to_text_numpy(curr_obs)
#         curr_text = "\n".join(curr_text_list)
#         text_obs.append(curr_text)

#     batch_tokens = tokenizer(
#         text_obs,
#         return_tensors="pt",
#         add_special_tokens=False,
#         padding=True,
#         truncation=True,
#     ).to(llm_pretrained.device)

#     with torch.no_grad():
#         hidden_states, _ = llm_pretrained.forward_hidden_states(**batch_tokens, use_cache=True, output_hidden_states=True, target_layer=[layer])

#     numpy_embed = embed.cpu().numpy().astype(np.float32)

#     del hidden_states
#     del batch_tokens
#     del embed
#     torch.cuda.empty_cache()

#     return numpy_embed

# llm_pretrained_raw = AutoModelForCausalLM.from_pretrained(
#     local_dir,
#     torch_dtype=torch.float16,
#     device_map="balanced_low_0",
#     low_cpu_mem_usage=True,
# )
# tokenizer = AutoTokenizer.from_pretrained(local_dir, device_map="balanced_low_0")

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({"pad_token": "<pad>"})
#     llm_pretrained_raw.resize_token_embeddings(len(tokenizer), mean_resizing=False)
#     embedding_dim = llm_pretrained_raw.get_input_embeddings().weight.shape[1]
#     padding_token_id = tokenizer.convert_tokens_to_ids("<pad>")

#     with torch.no_grad():
#         llm_pretrained_raw.get_input_embeddings().weight[
#             padding_token_id
#         ] = torch.zeros(embedding_dim)

# # llm_pretrained_raw = DDP(llm_pretrained_raw, device_ids=[device])
# # llm_pretrained = llm_pretrained_raw
# llm_pretrained = torch.compile(llm_pretrained_raw, fullgraph=True)

# def get_llm_obs(obs, embeddding_type, layer, k):
#     obs = np.array(obs)
#     text_obs = []
#     for curr_obs in obs:
#         curr_text_list = symbolic_to_text_numpy(curr_obs)
#         curr_text = "\n".join(curr_text_list)
#         text_obs.append(curr_text)
#     batch_tokens = tokenizer(
#         text_obs,
#         return_tensors="pt",
#         add_special_tokens=False,
#         padding=True,
#         truncation=True,
#     ).to(llm_pretrained.device)

#     with torch.no_grad():
#         hidden_states = llm_pretrained(**batch_tokens, output_hidden_states=True)[
#             "hidden_states"
#         ][layer]

#     if embeddding_type == 1:
#         embed = hidden_states.mean(axis=1)

#     else:
#         batch_size, seq_len, hidden_dim = hidden_states.shape
#         indices = batch_tokens["attention_mask"].sum(1)
#         safe_indices = torch.clamp(indices - k, min=0)
#         last_k_indices = safe_indices.unsqueeze(1) + torch.arange(
#             k, device=indices.device
#         ).unsqueeze(0)
#         embed = hidden_states[torch.arange(batch_size).unsqueeze(1), last_k_indices]
#         embed = embed.mean(axis=1)

#     numpy_embed = embed.cpu().numpy().astype(np.float32)

#     del hidden_states
#     del batch_tokens
#     del embed
#     torch.cuda.empty_cache()

#     return numpy_embed


# def get_llm_obs(obs, embeddding_type, layer, k):
#     batch_size = 64
#     obs = np.array(obs)
#     text_obs = []
#     for curr_obs in obs:
#         curr_text_list = symbolic_to_text_numpy(curr_obs)
#         curr_text = "\n".join(curr_text_list)
#         text_obs.append(curr_text)

#     numpy_embed = np.zeros((obs.shape[0], 4096), dtype=np.float32)
#     for i in range(obs.shape[0]//batch_size):
#         batch_tokens = tokenizer(text_obs[i*batch_size:(i+1)*batch_size], return_tensors="pt", add_special_tokens=False, padding=True, truncation=True).to(llm_pretrained.device)
#         with torch.no_grad():
#             hidden_states = llm_pretrained(**batch_tokens, output_hidden_states=True)["hidden_states"][layer]

#         if embeddding_type == 1:
#             embed = hidden_states.mean(axis=1)

#         else:
#             batch_size, seq_len, hidden_dim = hidden_states.shape
#             indices = batch_tokens['attention_mask'].sum(1)
#             safe_indices = torch.clamp(indices - k, min=0)
#             last_k_indices = safe_indices.unsqueeze(1) + torch.arange(k, device=indices.device).unsqueeze(0)
#             embed = hidden_states[torch.arange(batch_size).unsqueeze(1), last_k_indices]
#             embed = embed.mean(axis=1)

#         numpy_embed[i*batch_size:(i+1)*batch_size] = embed.cpu().numpy()

#     return numpy_embed
