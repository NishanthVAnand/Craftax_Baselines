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
parser.add_argument("--obs_only", type=int, default=1,
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
embeddings_diff = []
raw_obs = []
raw_obs_diff = []

for embed_type in []:
    config["EMB_TYPE"] = embed_type

    for i in range(128):
        old_obsvv = obsvv
        rng, rng_a, _rng = jax.random.split(rng, 3)
        action = jax.random.randint(
            rng_a, shape=(num_envs, ), minval=0, maxval=17)
        obsvv, env_state, reward_e, done, info = env.step(
            _rng, env_state, action, env_params
        )
        raw_obs.append(np.array(obsvv[0]))
        raw_obs_diff.append(np.array(obsvv[0] - old_obsvv[0]))

        return_dtype = jax.ShapeDtypeStruct(
            (config["NUM_ENVS"], config["NUM_PARAMS"]), jnp.float32)

        obsv_embed_dif = jax.pure_callback(
            get_llm_obs,
            return_dtype,
            jnp.abs(obsvv - old_obsvv),
            config["LAYER"],
            config["EMB_TYPE"],
            config["DECAY"],
            config["EQ_SPLIT"],
            config["OBS_TYPE"],
            config["OBS_ONLY"],)

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
        embeddings_diff.append(np.array(obsv_embed_dif[0]))

    emb_np = np.array(embeddings)
    raw_obs_np = np.array(raw_obs)
    emb_np_d = np.array(embeddings_diff)
    raw_obs_np_d = np.array(raw_obs_diff)

    sim_matrices = {
        "Raw": cosine_similarity(raw_obs_np),
        "Embed": cosine_similarity(emb_np),
        "Raw diff": cosine_similarity(raw_obs_np_d),
        "Embed diff": cosine_similarity(emb_np_d),
    }

    # Compute global vmin and vmax for shared color scale
    all_values = np.concatenate([m.flatten() for m in sim_matrices.values()])
    vmin = np.min(all_values)
    vmax = np.max(all_values)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for ax, (title, sim_matrix) in zip(axs, sim_matrices.items()):
        im = ax.imshow(sim_matrix, cmap='viridis',
                       interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Timestep")

    # Add a single colorbar on the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=f"Cosine Similarity for {embed_type}")

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(f"similarity_matrix_combined_{embed_type}.png", dpi=300)
    plt.close()
