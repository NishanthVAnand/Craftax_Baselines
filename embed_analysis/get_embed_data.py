from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from craftax.craftax_classic.constants import *
import numpy as np

import jax
from craftax.craftax_env import make_craftax_env_from_name
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
    RewardWrapper,
)
from rewards import *
from llm_observation_classic import get_llm_obs
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
parser.add_argument("--achievement", type=str, default="PLACE_TABLE")
parser.add_argument(
    "--total_timesteps", type=lambda x: int(float(x)), default=1e9
)  # Allow scientific notation
parser.add_argument("--seed", type=int)
parser.add_argument("--layer", type=int, nargs="+", default=[17])
parser.add_argument("--emb_type", type=int, default=0, help="0: mean, 1: exp")
parser.add_argument("--eq_split", type=int, default=16,
                    help="how many equal parts")
parser.add_argument("--decay", type=float, default=0.9)
parser.add_argument(
    "--obs_type", type=int, default=0, help="0: nearest only text, 1: all text, 2: all map"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=3,
)
parser.add_argument("--obs_only", type=int, default=0,
                    help="0: use all, 1: only obs")

parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
args = parser.parse_args()

if args.obs_only == 0:
    concat_size = 0
elif args.obs_only == 1:
    concat_size = 22
elif args.obs_only == 2:
    concat_size = 1345
else:
    raise ValueError(f"Args.obs_only {args.obs_only} not known")

hidden_size = 4096
emb_dict_map = {
    0: hidden_size * len(args.layer) + concat_size,
    1: hidden_size * len(args.layer) + concat_size,
    2: hidden_size * len(args.layer) + concat_size,
    3: hidden_size * int(args.eq_split) * len(args.layer) + concat_size,
    4: hidden_size * int(args.eq_split) * len(args.layer) + concat_size,
    5: hidden_size * len(args.layer) + concat_size,
    6: hidden_size * int(args.eq_split) * len(args.layer) + concat_size,
}
args.num_params = emb_dict_map[int(args.emb_type)]

config = vars(args)
for key in list(config.keys()):
    config[key.upper()] = config[key]
    del config[key]

num_envs = 3
env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', True)
env_params = env.default_params
achievement = "WAKE_UP"
env = RewardWrapper(env, achievement, get_basic_rewards(achievement))
env = LogWrapper(env)
env = OptimisticResetVecEnvWrapper(
    env,
    num_envs=num_envs,
    reset_ratio=min(4, num_envs),
)

rng = jax.random.PRNGKey(0)

rng, reset_rng = jax.random.split(rng)
obsvv, env_state = env.reset(reset_rng, env_params)

embeddings = []
raw_obs = []
for i in range(128):
    rng, rng_a, _rng = jax.random.split(rng, 3)
    action = jax.random.randint(rng_a, shape=(num_envs, ), minval=0, maxval=17)
    obsvv, env_state, reward_e, done, info = env.step(
        _rng, env_state, action, env_params
    )
    raw_obs.append(np.array(obsvv[0]))
    return_dtype = jax.ShapeDtypeStruct(
        (config["NUM_ENVS"], config["NUM_PARAMS"]), jnp.float32)
    obsv = jax.pure_callback(
        get_llm_obs,
        return_dtype,
        obsvv,
        config["LAYER"],
        config["EMB_TYPE"],
        config["DECAY"],
        config["EQ_SPLIT"],
        config["OBS_TYPE"],
        config["OBS_ONLY"],)

    embeddings.append(np.array(obsv[0]))
    if done[0]:
        break

# embeddings = jnp.concatenate(embeddings, axis=0)
raw_obs = jnp.concatenate(raw_obs, axis=0)

emb_np = np.array(embeddings)
raw_obs_np = np.array(raw_obs)

sim_matrix = cosine_similarity(emb_np)  # shape: (N, N)

# Plot and save the similarity heatmap
plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest')
plt.title("Cosine Similarity Matrix")
plt.xlabel("Timestep")
plt.ylabel("Timestep")
plt.colorbar(label="Cosine Similarity")
plt.tight_layout()
plt.savefig("similarity_matrix_emb.png", dpi=300)
plt.close()



sim_matrix = cosine_similarity(emb_np)  # shape: (N, N)

# Plot and save the similarity heatmap
plt.figure(figsize=(8, 6))
plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest')
plt.title("Cosine Similarity Matrix")
plt.xlabel("Timestep")
plt.ylabel("Timestep")
plt.colorbar(label="Cosine Similarity")
plt.tight_layout()
plt.savefig("similarity_matrix_raw.png", dpi=300)
plt.close()

# pca = PCA(n_components=2)
# emb_2d = pca.fit_transform(emb_np)
#
# plt.figure(figsize=(6, 5))
# plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
#             c=np.arange(len(emb_2d)), cmap='viridis')
# plt.title("PCA of embeddings over time")
# plt.colorbar(label='Timestep')
# plt.tight_layout()
# plt.savefig("embedding_pca.png", dpi=300)
# plt.close()
#
#
# sim = cosine_similarity(emb_np[:-1], emb_np[1:])
# step_sim = np.diag(sim)
#
# plt.figure(figsize=(6, 4))
# plt.plot(step_sim)
# plt.title("Cosine Similarity Between Consecutive Timesteps")
# plt.xlabel("Timestep")
# plt.ylabel("Cosine Similarity")
# plt.tight_layout()
# plt.savefig("embedding_cosine_similarity.png", dpi=300)
# plt.close()
#
#
# sim = cosine_similarity(raw_obs_np[:-1], raw_obs_np[1:])
# step_sim = np.diag(sim)
#
# plt.figure(figsize=(6, 4))
# plt.plot(step_sim)
# plt.title("Cosine Similarity Between Consecutive Timesteps")
# plt.xlabel("Timestep")
# plt.ylabel("Cosine Similarity")
# plt.tight_layout()
# plt.savefig("raw_obs_cosine_similarity.png", dpi=300)
# plt.close()
#
# norms = np.linalg.norm(emb_np, axis=1)
#
# plt.figure(figsize=(6, 4))
# plt.plot(norms)
# plt.title("L2 Norm of Embeddings Over Time")
# plt.xlabel("Timestep")
# plt.ylabel("L2 Norm")
# plt.tight_layout()
# plt.savefig("embedding_l2_norms.png", dpi=300)
# plt.close()
