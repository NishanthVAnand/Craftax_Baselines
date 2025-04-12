from craftax.craftax_classic.constants import *
import numpy as np

Block_id_to_text = {
    0: "INVALID",
    1: "OUT_OF_BOUNDS",
    2: "GRASS",
    3: "WATER",
    4: "STONE",
    5: "TREE",
    6: "WOOD",
    7: "PATH",
    8: "COAL",
    9: "IRON",
    10: "DIAMOND",
    11: "CRAFTING_TABLE",
    12: "FURNACE",
    13: "SAND",
    14: "LAVA",
    15: "PLANT",
    16: "RIPE_PLANT",
}

mob_id_to_text = {
    0: "Zombie",
    1: "Cow",
    2: "Skeleton",
    3: "Arrow",
}

Inventory_Items = [
    "Wood",
    "Stone",
    "Coal",
    "Iron",
    "Diamond",
    "Sapling",
    "Wooden Pickaxe",
    "Stone Pickaxe",
    "Iron Pickaxe",
    "Wooden Sword",
    "Stone Sword",
    "Iron Sword",
]

Intrinsic_Items = [
    "Health",
    "Food",
    "Drink",
    "Energy",
]
Direction = ["Left", "Right", "Up", "Down"]
Direction_to_obj = {
    0: (OBS_DIM[0] // 2, OBS_DIM[1] // 2 - 1),
    1: (OBS_DIM[0] // 2, OBS_DIM[1] // 2 + 1),
    2: (OBS_DIM[0] // 2 - 1, OBS_DIM[1] // 2),
    3: (OBS_DIM[0] // 2 + 1, OBS_DIM[1] // 2),
}

ACHIEVEMENTS = [
    "Collect Wood",
    "Place Table",
    "Eat Cow",
    "Collect Sampling",
    "Collect Drink",
    "Make Wood Pickaxe",
    "Make Wood Sword",
    "Place Plant",
    "Defeat Zombie",
    "Collect Stone",
    "Place Stone",
    "Eat Plant",
    "Defeat Skeleton",
    "Make Stone Pickaxe",
    "Make Stone Sword",
    "Wake Up",
    "Place Furnace",
    "Collect Coal",
    "Collect Iron",
    "Make Iron Pickaxe",
    "Make Iron Sword",
    "Collect Diamond",
]


def generate_distance_dict(max_range=5):
    """Precompute all possible movement descriptions within a given range."""
    distance_dict = {}
    for v in range(-max_range, max_range + 1):
        for h in range(-max_range, max_range + 1):
            if v == 0 and h == 0:
                continue  # Skip (0,0) since it means no movement

            # Vertical description
            vert_text = f"{abs(v)} steps north" if v < 0 else (f"{v} steps south" if v > 0 else "")

            # Horizontal description
            hor_text = f"{abs(h)} steps west" if h < 0 else (f"{h} steps east" if h > 0 else "")

            # Combine descriptions
            if vert_text and hor_text:
                distance_dict[(v, h)] = vert_text + " and " + hor_text
            else:
                distance_dict[(v, h)] = vert_text or hor_text

    return distance_dict


distance_lookup = generate_distance_dict(
    max_range=5
)  # precompute all possible movement descriptions within a given range


def symbolic_to_text_numpy(symbolic_array):

    text_description = []
    meta_prompt = "You are an intelligent agent exploring the world of Crafter — a procedurally generated open-ended survival game. "
    meta_prompt += "It is a 2D tile-based environment with nearby tiles visible to you. "
    meta_prompt += "Your goal is to survive, gather resources, and explore. You should also complete achievements (eg. sleep) to get rewards. "
    meta_prompt += "You will receive an observation describing your current view, which contain various sections such as blocks (grass, sand, etc), items (torch, ladder), "
    meta_prompt += "mobs (zombie, cow, arrow, etc), inventory (wood, iron, diamond, etc), intrinsic values (health, food, drink, and energy). "
    meta_prompt += "Maintaining high levels are important for the agent to stay alive. "
    meta_prompt += "You will also receive the brightness level of the environment indicating the time of the day. "
    meta_prompt += "Your task is to interpret and remember the details of this observation.\n"
    meta_prompt += (
        "The agent will then use this information to complete the following achievements: "
    )
    meta_prompt += ", ".join(ACHIEVEMENTS) + ". "
    text_description.append(meta_prompt)

    rows = np.arange(OBS_DIM[0])[:, None]
    cols = np.arange(OBS_DIM[1])[None, :]
    distance_matrix = np.abs(rows - (OBS_DIM[0] - 1) // 2) + np.abs(cols - (OBS_DIM[1] - 1) // 2)
    max_distance = 100  # max distance to get rid of zeros

    symbolic_array_map = symbolic_array[:1323]
    symbolic_array_map_reshaped = symbolic_array_map.reshape(OBS_DIM[0], OBS_DIM[1], -1)

    # Block types description
    symbolic_array_map_blocks = symbolic_array_map_reshaped[:, :, :17]
    if symbolic_array_map_blocks.sum() > 0:
        block_description = "There are a total of 17 different types of blocks: "
        block_description += ", ".join([Block_id_to_text[i] for i in range(17)])
        block_description += ". The following blocks appear in your sight: "
        text_description.append(block_description)
        block_types = np.argmax(symbolic_array_map_blocks, axis=-1)
        unique_blocks = np.unique(block_types)
        unique_blocks = unique_blocks[~np.isin(unique_blocks, [0, 1])]
        for block in unique_blocks:
            curr_block_mask = block_types == block
            curr_block_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
            curr_blocks = distance_matrix * curr_block_mask
            curr_blocks_max_dist = np.where(curr_block_mask, curr_blocks, max_distance)
            min_distance_curr_block = np.min(curr_blocks_max_dist)
            min_dist_indices = np.argwhere(curr_blocks_max_dist == min_distance_curr_block)
            relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
            distance_tuples = [tuple(map(int, d)) for d in relative_pos]
            descriptions = [distance_lookup.get(d, "Unknown movement") for d in distance_tuples]
            text_description.append(Block_id_to_text[block] + " is at " + ", ".join(descriptions))

    # Direction
    direction_array = symbolic_array[1339:1343]
    # text_description.append(
    #     "The agent is facing " + Direction[np.argwhere(direction_array == 1).item()]
    # )
    text_description.append(
        "The agent is facing "
        + Block_id_to_text[
            np.argmax(symbolic_array_map_blocks, axis=-1)[
                Direction_to_obj[np.argwhere(direction_array == 1).item()]
            ]
        ]
    )

    # Mob types description
    symbolic_array_map_mobs = symbolic_array_map_reshaped[:, :, 17:21]
    if symbolic_array_map_mobs.sum() > 0:
        mob_types = np.argmax(symbolic_array_map_mobs, axis=-1)
        mob_description = "There are a total of 4 different types of mobile objects: "
        mob_description += ", ".join([mob_id_to_text[i] for i in range(4)])
        mob_description += ". The following mobile objects appear in your sight: "
        text_description.append(mob_description)
        unique_mobs = np.unique(mob_types)
        for mob in unique_mobs:
            curr_mob_mask = mob_types == mob
            curr_mob_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
            curr_mobs = distance_matrix * curr_mob_mask
            curr_mobs_max_dist = np.where(curr_mob_mask, curr_mobs, max_distance)
            min_distance_curr_mob = np.min(curr_mobs_max_dist)
            min_dist_indices = np.argwhere(curr_mobs_max_dist == min_distance_curr_mob)
            relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
            distance_tuples = [tuple(map(int, d)) for d in relative_pos]
            descriptions = [distance_lookup.get(d, "Unknown movement") for d in distance_tuples]
            text_description.append(mob_id_to_text[mob] + " is at: " + ", ".join(descriptions))

    inventory_array = np.argwhere(symbolic_array[1323:1335] > 0).flatten()
    if inventory_array.size > 0:
        inventory_description = (
            "The agent can store a total of 12 different types of items in its inventory: "
        )
        inventory_description += ", ".join([Inventory_Items[i] for i in range(12)])
        inventory_description += ". You currently have the following items in your inventory: "
        text_description.append(inventory_description)
        for inv_idx in inventory_array:
            item_count = symbolic_array[1323:1335][inv_idx]
            text_description.append(
                "The agent has " + str(item_count) + " units of " + Inventory_Items[inv_idx]
            )

    intrinsic_array = symbolic_array[1335:1339]
    text_description.append(
        "Below are the intrinsic values of the agent that describes its condition: "
    )
    for intrinsic_idx in range(len(Intrinsic_Items)):
        intrinsic_value = intrinsic_array[intrinsic_idx]
        desc = ""
        if intrinsic_value < 0.5:
            desc = " The agent's " + Intrinsic_Items[intrinsic_idx] + " is critical!"
        text_description.append(
            Intrinsic_Items[intrinsic_idx]
            + " is "
            + str(np.ceil(intrinsic_value * 10))
            + "/10."
            + desc
        )

    text_description.append(
        "The light level is " + str(np.around(symbolic_array[1343] * 100, 2)) + "%."
    )

    if symbolic_array[1344] == 1:
        text_description.append("The agent is sleeping.")
    else:
        text_description.append("The agent is awake.")

    return text_description
