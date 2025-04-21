from craftax.craftax_classic.constants import *
import numpy as np

all_block_types_neutral = np.array(
    [
        "ZENTH",  # INVALID
        "VORIN",  # OUT_OF_BOUNDS
        "MIRAQ",  # GRASS
        "SYNEX",  # WATER
        "DRAEL",  # STONE
        "TREXO",  # TREE
        "WODEN",  # WOOD
        "TRAKO",  # PATH
        "COBRAX",  # COAL
        "FERON",  # IRON
        "LUMEX",  # DIAMOND
        "CRAFIX",  # CRAFTING_TABLE
        "FURNIQ",  # FURNACE
        "SANDRA",  # SAND
        "LAVOS",  # LAVA
        "PLENTH",  # PLANT
        "RIPEL",  # RIPE_PLANT
    ]
)

all_mob_types_neutral = np.array(
    [
        "",  # (empty)
        "CRAVOX",  # CRAVOX (kept for flair)
        "BOVRA",  # COW
        "SKELOM",  # SKELETON
        "VOLTRA",  # ARROW
    ]
)

Inventory_Items_neutral = [
    "COREBRICK",  # WOOD
    "HARDNODE",  # STONE
    "DARKDUST",  # COAL
    "FERROCORE",  # IRON
    "LUMENCRYST",  # DIAMOND
    "SPAWNSTEM",  # SAPLING
    "GEARPICK",  # WOODEN PICKAXE
    "STRIKECORE",  # STONE PICKAXE
    "METAPICK",  # IRON PICKAXE
    "SLASHFORM",  # WOODEN SWORD
    "EDGELITH",  # STONE SWORD
    "IRONSPIKE",  # IRON SWORD
]

all_block_types = np.array(
    [
        "INVALID",
        "OUT_OF_BOUNDS",
        "GRASS",
        "WATER",
        "STONE",
        "TREE",
        "WOOD",
        "PATH",
        "COAL",
        "IRON",
        "DIAMOND",
        "CRAFTING_TABLE",
        "FURNACE",
        "SAND",
        "LAVA",
        "PLANT",
        "RIPE_PLANT",
    ]
)
all_mob_types = np.array(["", "ZOMBIE", "COW", "SKELETON", "ARROW"])

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
    0: "",
    1: "ZOMBIE",
    2: "COW",
    3: "SKELETON",
    4: "ARROW",
}

Inventory_Items = [
    "WOOD",
    "STONE",
    "COAL",
    "IRON",
    "DIAMOND",
    "SAPLING",
    "WOODEN PICKAXE",
    "STONE PICKAXE",
    "IRON PICKAXE",
    "WOODEN SWORD",
    "STONE SWORD",
    "IRON SWORD",
]

Intrinsic_Items = [
    "Health",
    "Food",
    "Hydration",
    "Wakefulness",
]

Direction = ["Left", "Right", "Up", "Down"]

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


def symbolic_to_text_numpy(symbolic_array, obs_type=0, obs_only=0, crop_size=0):
    """
    obs_type: 0 for nearest only
    obs_type: 1 for all
    obs_type: 2 for map view
    obs_type: 3 for map view with neutral names
    obs_type: 4 for map view with detailed description on where the agent can reach in N steps
    obs_only: False then intrisincs and inventory are included
    obs_only: True then only observation is included
    """

    text_description = []
    meta_prompt = "You are an intelligent agent exploring the world of Crafter — a procedurally generated open-ended survival game. "
    meta_prompt += "It is a 2D tile-based environment with nearby tiles visible to you. "
    # meta_prompt += "Your goal is to survive, gather resources, and explore. You should also complete achievements (eg. sleep) to get rewards. "
    # meta_prompt += "You will receive an observation describing your current view, which contain various sections such as blocks (grass, sand, etc), items (torch, ladder), "
    # meta_prompt += "mobs (zombie, cow, arrow, etc), inventory (wood, iron, diamond, etc), intrinsic values (health, food, drink, and energy). "
    # meta_prompt += "Maintaining high levels are important for the agent to stay alive. "
    # meta_prompt += "You will also receive the brightness level of the environment indicating the time of the day. "
    meta_prompt += "Your task is to interpret this observation, store important features that is helpful to the agent, "
    meta_prompt += "and come up with a strategy to help the agent achieve its accomplishments \n"
    meta_prompt += "The agent has the below list of achievements to complete: "
    meta_prompt += ", ".join(ACHIEVEMENTS) + ". "
    text_description.append(meta_prompt)

    if crop_size > 0:
        OBS_DIM = (crop_size, crop_size)
    else:
        OBS_DIM = (7, 9)

    Direction_to_obj = {
        0: (OBS_DIM[0] // 2, OBS_DIM[1] // 2 - 1),
        1: (OBS_DIM[0] // 2, OBS_DIM[1] // 2 + 1),
        2: (OBS_DIM[0] // 2 - 1, OBS_DIM[1] // 2),
        3: (OBS_DIM[0] // 2 + 1, OBS_DIM[1] // 2),
    }

    rows = np.arange(OBS_DIM[0])[:, None]
    cols = np.arange(OBS_DIM[1])[None, :]
    distance_matrix = np.abs(rows - (OBS_DIM[0] - 1) // 2) + np.abs(cols - (OBS_DIM[1] - 1) // 2)
    max_distance = 100  # max distance to get rid of zeros

    symbolic_array_map = symbolic_array[:-22]
    if crop_size > 0:
        symbolic_array_map_reshaped = symbolic_array_map.reshape(OBS_DIM[0], OBS_DIM[1], -1)
    else:
        symbolic_array_map_reshaped = symbolic_array_map.reshape(OBS_DIM[0], OBS_DIM[1], -1)
    symbolic_array_map_blocks = symbolic_array_map_reshaped[:, :, :17]
    symbolic_array_map_mobs = symbolic_array_map_reshaped[:, :, 17:21]
    symbolic_array_map_mobs = np.concatenate(
        [np.zeros((OBS_DIM[0], OBS_DIM[1], 1)), symbolic_array_map_mobs], axis=-1
    )

    if obs_type in [2, 4, 5]:
        block_description = "There are a total of 16 different types of blocks: "
        block_description += ", ".join(all_block_types[1:])
        text_description.append(block_description)

        mob_description = "There are a total of 4 different types of mobile objects: "
        mob_description += ", ".join(all_mob_types[1:])
        text_description.append(mob_description)

        dependency_text = (
            "The dependency chart is as follows: Trees -> Wood -> Crafting table -> Wooden tools. "
        )
        dependency_text += "Wooden pickaxe -> Stone -> Stone tools and Furnace. "
        dependency_text += "Furnace + Coal -> Smelt iron -> Iron tools. "
        dependency_text += "Iron pickaxe -> Diamonds. "
        dependency_text += "Parallel loops: "
        dependency_text += "Farming (Plant -> Ripe_Plant -> Eat). "
        dependency_text += "Food Cow and Drink. "
        dependency_text += "Combat achievements (zombie, skeleton). "
        text_description.append(dependency_text)

        grid_description = "Below is the observation that is visible to the agent. "
        grid_description += "This is a {}x{} grid, where each cell describes a combination of block type and a mobile object type (if present). ".format(
            OBS_DIM[0], OBS_DIM[1]
        )
        grid_description += "The map is organized in rows and columns, and each cell contains a string in the format: <block type> and <mobile object type>. "
        grid_description += (
            "The grid is ordered row by row, from top to bottom, and from left to right. "
        )
        grid_description += (
            "Each row represents a horizontal slice of the map and the rows are separated by @. "
        )
        text_description.append(grid_description)

        block_types = np.argmax(symbolic_array_map_blocks, axis=-1)
        mob_types = np.argmax(symbolic_array_map_mobs, axis=-1)

        block_types_str = all_block_types[block_types]
        block_types_str[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = "Agent"
        mob_types_str = all_mob_types[mob_types]

        both_types = np.where(
            mob_types_str != "",
            np.char.add(
                np.char.add(block_types_str.astype(str), " and "), mob_types_str.astype(str)
            ),
            block_types_str,
        )

        text_description.append("\n".join(["@ " + " | ".join(row) + " @" for row in both_types]))

        if obs_type == 4:
            text_description.append(
                get_obs_type_4_description(block_types_str, mob_types_str, distance_matrix)
            )

    elif obs_type == 3:
        block_description = "There are a total of 16 different types of blocks: "
        block_description += ", ".join(all_block_types_neutral)
        text_description.append(block_description)

        mob_description = "There are a total of 4 different types of mobile objects: "
        mob_description += ", ".join(all_mob_types_neutral)
        text_description.append(mob_description)

        grid_description = "Below is the observation that is visible to the agent. "
        grid_description += "This is a 7x9 grid, where each cell describes a combination of block type and a mobile object type (if present). "
        grid_description += "The map is organized in rows and columns, and each cell contains a string in the format: <block type> and <mobile object type>. "
        grid_description += (
            "The grid is ordered row by row, from top to bottom, and from left to right. "
        )
        grid_description += (
            "Each row represents a horizontal slice of the map and the rows are separated by @. "
        )
        text_description.append(grid_description)

        block_types = np.argmax(symbolic_array_map_blocks, axis=-1)
        mob_types = np.argmax(symbolic_array_map_mobs, axis=-1)

        block_types_str = all_block_types_neutral[block_types]
        block_types_str[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = "Agent"
        mob_types_str = all_mob_types_neutral[mob_types]

        both_types = np.where(
            mob_types_str != "",
            np.char.add(
                np.char.add(block_types_str.astype(str), " and "), mob_types_str.astype(str)
            ),
            block_types_str,
        )

        text_description.append("\n".join(["@ " + " | ".join(row) + " @" for row in both_types]))

    else:
        # Block types description
        if symbolic_array_map_blocks.sum() > 0:
            block_description = "There are a total of 16 different types of blocks: "
            block_description += ", ".join([Block_id_to_text[i + 1] for i in range(16)])
            block_description += ". The following blocks appear in your sight: "
            text_description.append(block_description)
            block_types = np.argmax(symbolic_array_map_blocks, axis=-1)
            unique_blocks = np.unique(block_types)
            unique_blocks = unique_blocks[~np.isin(unique_blocks, [1])]
            for block in unique_blocks:
                curr_block_mask = block_types == block
                curr_block_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
                curr_blocks = distance_matrix * curr_block_mask
                if obs_type == 0:
                    curr_blocks_max_dist = np.where(curr_block_mask, curr_blocks, max_distance)
                    min_distance_curr_block = np.min(curr_blocks_max_dist)
                    min_dist_indices = np.argwhere(curr_blocks_max_dist == min_distance_curr_block)
                    relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
                elif obs_type == 1:
                    relative_pos = np.argwhere(curr_blocks > 0) - np.array(
                        [OBS_DIM[0] // 2, OBS_DIM[1] // 2]
                    )
                distance_tuples = [tuple(map(int, d)) for d in relative_pos]
                descriptions = [distance_lookup.get(d, "Unknown movement") for d in distance_tuples]
                text_description.append(
                    Block_id_to_text[block] + " is at " + ", ".join(descriptions)
                )

        # Mob types description
        if symbolic_array_map_mobs.sum() > 0:
            mob_types = np.argmax(symbolic_array_map_mobs, axis=-1)
            mob_description = "There are a total of 4 different types of mobile objects: "
            mob_description += ", ".join([mob_id_to_text[i + 1] for i in range(4)])
            mob_description += ". The following mobile objects appear in your sight: "
            text_description.append(mob_description)
            unique_mobs = np.unique(mob_types)
            unique_mobs = unique_mobs[~np.isin(unique_mobs, [0])]
            for mob in unique_mobs:
                curr_mob_mask = mob_types == mob
                curr_mob_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
                curr_mobs = distance_matrix * curr_mob_mask
                if obs_type == 0:
                    curr_mobs_max_dist = np.where(curr_mob_mask, curr_mobs, max_distance)
                    min_distance_curr_mob = np.min(curr_mobs_max_dist)
                    min_dist_indices = np.argwhere(curr_mobs_max_dist == min_distance_curr_mob)
                    relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
                elif obs_type == 1:
                    relative_pos = np.argwhere(curr_mobs > 0) - np.array(
                        [OBS_DIM[0] // 2, OBS_DIM[1] // 2]
                    )
                distance_tuples = [tuple(map(int, d)) for d in relative_pos]
                descriptions = [distance_lookup.get(d, "Unknown movement") for d in distance_tuples]
                text_description.append(mob_id_to_text[mob] + " is at: " + ", ".join(descriptions))

    # Direction
    direction_array = symbolic_array[-6:-2]
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

    if obs_only:
        return text_description

    inventory = np.round((symbolic_array[-22:-10] * 10) ** 2).astype(np.int64)
    inventory_description = (
        "The agent can store a total of 12 different types of items in its inventory: "
    )
    if obs_type == 3:
        inventory_description += (
            ", ".join([Inventory_Items_neutral[i].upper() for i in range(12)]) + "."
        )
    else:
        inventory_description += ", ".join([Inventory_Items[i].upper() for i in range(12)]) + "."
    text_description.append(inventory_description)

    if inventory.sum() > 0:
        inventory_array = np.argwhere(inventory > 0).flatten()
        inventory_description = "You currently have the following items in your inventory: "
        text_description.append(inventory_description)
        for inv_idx in inventory_array:
            item_count = inventory[inv_idx]
            if obs_type == 3:
                text_description.append(
                    "The agent has "
                    + str(item_count)
                    + " units of "
                    + Inventory_Items_neutral[inv_idx]
                )
            else:
                text_description.append(
                    "The agent has " + str(item_count) + " units of " + Inventory_Items[inv_idx]
                )

    intrinsic_array = symbolic_array[-10:-6]
    text_description.append(
        "Below are the intrinsic values of the agent that describes its condition. Maintaining high levels are important for the agent to stay alive."
    )
    for intrinsic_idx in range(len(Intrinsic_Items)):
        intrinsic_value = intrinsic_array[intrinsic_idx]
        desc = ""
        if intrinsic_value < 0.5:
            desc = " The agent's " + Intrinsic_Items[intrinsic_idx] + " is critical!"
        text_description.append(
            Intrinsic_Items[intrinsic_idx]
            + " is "
            + str(round(intrinsic_value * 10))
            + "/10."
            + desc
        )

    text_description.append(
        "The brightness level of the environment indicates the time of the day. The brightness level is "
        + str(np.around(symbolic_array[-2] * 100, 2))
        + "%."
    )

    if symbolic_array[-1] == 1:
        text_description.append("The agent is sleeping.")
    else:
        text_description.append("The agent is awake.")

    return text_description


resources_list = ["WATER", "STONE", "TREE", "WOOD", "PLANT", "RIPE_PLANT", "COW"]
hazard_list = ["LAVA", "ZOMBIE", "SKELETON"]
mineral_list = ["COAL", "IRON", "DIAMOND"]
tool_maker_list = ["CRAFTING_TABLE", "FURNACE"]


def get_obs_type_4_description(block_types_str, mob_types_str, distance_matrix):
    nearest_distances = [1, 2, 3]
    additional_text = "Based on the observation provided, below are the details "
    additional_text += "of the nearest resources, minerals, hazards, and tool makers: \n"
    for distance in reversed(nearest_distances):
        curr_blocks = block_types_str[distance_matrix == distance]
        curr_mobs = mob_types_str[distance_matrix == distance]

        unique_blocks = np.unique(curr_blocks)
        unique_mobs = np.unique(curr_mobs)

        unique_all = np.concatenate((unique_blocks, unique_mobs))

        additional_text += "At distance " + str(distance) + ", "

        curr_resources = np.intersect1d(unique_all, resources_list)
        if len(curr_resources) > 0:
            additional_text += "Resources are: " + ", ".join(curr_resources) + ". "
        else:
            additional_text += "No resources found. "

        curr_hazards = np.intersect1d(unique_all, hazard_list)
        if len(curr_hazards) > 0:
            additional_text += "Hazards are: " + ", ".join(curr_hazards) + ". "
        else:
            additional_text += "No hazards found. "

        curr_tool_makers = np.intersect1d(unique_all, tool_maker_list)
        if len(curr_tool_makers) > 0:
            additional_text += "Tool makers are: " + ", ".join(curr_tool_makers) + ". "
        else:
            additional_text += "No tool makers found. "

        curr_minerals = np.intersect1d(unique_all, mineral_list)
        if len(curr_minerals) > 0:
            additional_text += "Minerals are: " + ", ".join(curr_minerals) + ". "
        else:
            additional_text += "No minerals found. "

    additional_text += "This information can help the agent make decisions about its next actions."
    return additional_text
