import torch
import numpy as np

from text_wrapper_crafter_classic import *
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_llama import CustomLlamaForCausalLM
from concurrent.futures import ThreadPoolExecutor

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

num_gpus = torch.cuda.device_count()
local_dir = "/home/s/saminur/scratch/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/"

llm_pretrained_all = [
    CustomLlamaForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
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
        llm_pretrained.get_input_embeddings().weight[padding_token_id] = torch.zeros(embedding_dim)

llm_pretrained_all = [torch.compile(llm_pretrained_all[i]) for i in range(num_gpus - 1)]

emb_dict_map = {0: "mean", 1: "exp", 2: "last-10", 3: "last-k", 4: "eq-k", 5: "max", 6: "geom-k"}


def gpu_inference(i, text_obs_chunk, layer, emb_type, decay, eq_split):
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
            target_layer=layer,
            emb_type=emb_dict_map[emb_type],
            decay=decay,
            eq_split=eq_split,
        )
    return hidden_states


def get_llm_obs(obs, layer, emb_type, decay, eq_split, obs_type, obs_only):
    obs = np.array(obs)
    layer = [int(lay) for lay in layer]
    emb_type = int(emb_type)
    decay = float(decay)
    eq_split = int(eq_split)
    obs_type = int(obs_type)
    obs_only = int(obs_only)

    text_obs = []
    for curr_obs in obs:
        curr_text_list = symbolic_to_text_numpy(
            symbolic_array=curr_obs, obs_type=obs_type, obs_only=obs_only
        )
        curr_text = "\n".join(curr_text_list)
        text_obs.append(curr_text)

    text_obs_chunks = [text_obs[i :: num_gpus - 1] for i in range(num_gpus - 1)]

    embed = []
    with ThreadPoolExecutor(max_workers=num_gpus - 1) as executor:
        futures = [
            executor.submit(gpu_inference, i, text_obs_chunks[i], layer, emb_type, decay, eq_split)
            for i in range(num_gpus - 1)
        ]
        for future in futures:
            embed.append(future.result().cpu().numpy().astype(np.float32))

    numpy_embed = np.concatenate(embed, axis=0)
    if obs_only:
        numpy_embed = np.concatenate([numpy_embed, obs[:, 1323:]], axis=1)

    torch.cuda.empty_cache()

    return numpy_embed
