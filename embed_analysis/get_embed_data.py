import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from llm_observation_classic import get_llm_obs
from text_wrapper_crafter_classic import symbolic_to_text_numpy
from wrappers import OptimisticResetVecEnvWrapper
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
parser.add_argument("--achievement", type=str, default="PLACE_TABLE")
parser.add_argument(
    "--total_timesteps", type=lambda x: int(float(x)), default=1e9
)  # Allow scientific notation
parser.add_argument("--seed", type=int)
parser.add_argument("--layer", type=int, nargs="+", default=17)
parser.add_argument("--emb_type", type=int, default=0, help="0: mean, 1: exp")
parser.add_argument("--eq_split", type=int, default=16,
                    help="how many equal parts")
parser.add_argument(
    "--obs_type", type=int, default=0, help="0: nearest only text, 1: all text, 2: all map"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
)
parser.add_argument("--obs_only", type=int, default=0,
                    help="0: use all, 1: only obs")

# parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

config = vars(parser.parse_args())
for key in list(config.keys()):
    config[key.upper()] = config[key]
    del config[key]


rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)


env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=True)
# env = OptimisticResetVecEnvWrapper(
#     env,
#     num_envs=config["NUM_ENVS"],
#     reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
# )
env_params = env.default_params


obs, state = env.reset(reset_rng, env_params)


for step_i in range(10):
    rng, action_rng, step_rng = jax.random.split(rng, 3)

    action = env.action_space(env_params).sample(action_rng)

    obs, state, reward, done, info = env.step(
        step_rng, state, action, env_params)

    obs = obs.reshape(1, -1)

    obs = np.array(obs)

    text_obs = []
    for curr_obs in obs:
        curr_text_list = symbolic_to_text_numpy(
            symbolic_array=curr_obs, obs_type=config['OBS_TYPE'], obs_only=config["OBS_ONLY"],
        )
        curr_text = "\n".join(curr_text_list)
        text_obs.append(curr_text)

    return_dtype = jax.ShapeDtypeStruct(
        (config["NUM_ENVS"], config["NUM_PARAMS"]), jnp.float32)
    obsv = jax.pure_callback(
        get_llm_obs,
        return_dtype,
        obs,
        config["LAYER"],
        config["EMB_TYPE"],
        config["DECAY"],
        config["EQ_SPLIT"],
        config["OBS_TYPE"],
        config["OBS_ONLY"],)


    breakpoint()

    print(f"Step {step_i}: reward={reward}, done={done}")

    if done:
        print("Episode ended early. Resetting...")
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, env_params)
