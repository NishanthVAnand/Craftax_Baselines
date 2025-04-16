from craftax.craftax_classic.constants import Achievement
import jax.numpy as jnp


def get_basic_rewards(achievement):
    if achievement == "PLACE_TABLE":
        basic_rewards = jnp.array(
            [
                Achievement["WAKE_UP"].value,
                Achievement["EAT_COW"].value,
                Achievement["COLLECT_DRINK"].value,
                Achievement["COLLECT_WOOD"].value,
            ],
            dtype=jnp.int32,
        )
    elif achievement == "EAT_PLANT":
        basic_rewards = jnp.array(
            [
                Achievement["WAKE_UP"].value,
                Achievement["EAT_COW"].value,
                Achievement["COLLECT_DRINK"].value,
                Achievement["SAPLING"].value,
                Achievement["PLACE_PLANT"].value,
            ],
            dtype=jnp.int32,
        )
    elif achievement == "MAKE_WOOD_PICKAXE":
        basic_rewards = jnp.array(
            [
                Achievement["WAKE_UP"].value,
                Achievement["EAT_COW"].value,
                Achievement["COLLECT_DRINK"].value,
                Achievement["COLLECT_WOOD"].value,
                Achievement["PLACE_TABLE"].value,
            ],
            dtype=jnp.int32,
        )
    elif achievement == "MAKE_WOOD_SWORD":
        basic_rewards = jnp.array(
            [
                Achievement["WAKE_UP"].value,
                Achievement["EAT_COW"].value,
                Achievement["COLLECT_DRINK"].value,
                Achievement["COLLECT_WOOD"].value,
                Achievement["PLACE_TABLE"].value,
            ],
            dtype=jnp.int32,
        )
    else:
        basic_rewards = jnp.array([], dtype=jnp.int32)

    return basic_rewards
