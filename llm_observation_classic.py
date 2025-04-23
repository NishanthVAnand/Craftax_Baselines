import torch
import numpy as np

from text_wrapper_crafter_classic import *
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_llama import CustomLlamaForCausalLM
from concurrent.futures import ThreadPoolExecutor

from normalization import MinMaxScaler, StandardScaler

import os

min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

num_gpus = torch.cuda.device_count()
local_dir = "/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct/"

tokenizer = AutoTokenizer.from_pretrained(local_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


def load_model(i):
    return CustomLlamaForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": i},
    ).eval()


# with ThreadPoolExecutor(max_workers=num_gpus - 1) as executor:
#     llm_pretrained_all = [executor.submit(load_model, i) for i in range(num_gpus - 1)]
# llm_pretrained_all = [torch.compile(llm_pretrained_all[i]) for i in range(num_gpus - 1)]

llm_pretrained_all = [torch.compile(load_model(i)) for i in range(1, num_gpus)]

emb_dict_map = {0: "mean", 1: "exp", 2: "last-10", 3: "last-k", 4: "eq-k", 5: "max", 6: "geom-k"}


def gpu_inference(i, text_obs_chunk, layer, emb_type, decay, eq_split, obs_type):
    if obs_type == 5:
        model_output = llm_pretrained_all[i].generate(
            **text_obs_chunk, cache_implementation="static", max_length=1000
        )
        model_output_text = tokenizer.decode(model_output[0], skip_special_tokens=True)
        text_obs_chunk = [
            text_chunk + model_chunk
            for text_chunk, model_chunk in zip(text_obs_chunk, model_output_text)
        ]

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


def get_llm_obs(obs, layer, emb_type, decay, eq_split, obs_type, obs_only, crop_size, norm_type=0):
    obs = np.array(obs)
    layer = [int(lay) for lay in layer]
    emb_type = int(emb_type)
    decay = float(decay)
    eq_split = int(eq_split)
    obs_type = int(obs_type)
    obs_only = int(obs_only)
    crop_size = int(crop_size)
    norm_type = int(norm_type)

    text_obs = []
    for curr_obs in obs:
        curr_text_list = symbolic_to_text_numpy(
            symbolic_array=curr_obs, obs_type=obs_type, obs_only=obs_only, crop_size=crop_size
        )
        curr_text = "\n".join(curr_text_list)
        text_obs.append(curr_text)

    if obs_type == 4:
        text_obs = get_obs_type_4_description(text_obs)

    text_obs_chunks = [text_obs[i :: num_gpus - 1] for i in range(num_gpus - 1)]

    embed = []
    with ThreadPoolExecutor(max_workers=num_gpus - 1) as executor:
        futures = [
            executor.submit(
                gpu_inference, i, text_obs_chunks[i], layer, emb_type, decay, eq_split, obs_type
            )
            for i in range(num_gpus - 1)
        ]
        for future in futures:
            embed.append(future.result().to(dtype=torch.float32).cpu().numpy())

    numpy_embed = np.concatenate(embed, axis=0)

    if norm_type == 0:
        pass
    elif norm_type == 1:
        numpy_embed = standard_scaler.transform(numpy_embed)
    elif norm_type == 2:
        numpy_embed = min_max_scaler.transform(numpy_embed)
    else:
        raise ValueError("Invalid normalization type. Choose 0, 1, or 2.")

    if obs_only:
        numpy_embed = np.concatenate([numpy_embed, obs[:, -22:]], axis=1)

    torch.cuda.empty_cache()

    return numpy_embed
