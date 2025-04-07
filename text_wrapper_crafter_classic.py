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
    1: "Gnome Warrior",
    2: "Orc Soldier",
    3: "Lizard",
    4: "Knight",
    5: "Troll",
    6: "Pigman",
    7: "Frost Troll",
    8: "Cow",
    9: "Bat",
    10: "Snail",
    16: "Skeleton",
    17: "Gnome Archer",
    18: "Orc Mage",
    19: "Kobold",
    20: "Archer",
    21: "Deep Thing",
    22: "Fire Elemental",
    23: "Ice Elemental",
    24: "Arrow",
    25: "Dagger",
    26: "Fireball",
    27: "Iceball",
    28: "Arrow",
    29: "Slimeball",
    30: "Fireball",
    31: "Iceball",
    32: "Arrow (Player)",
    33: "Dagger (Player)",
    34: "Fireball (Player)",
    35: "Iceball (Player)",
    36: "Arrow (Player)",
    37: "Slimeball (Player)",
    38: "Fireball (Player)",
    39: "Iceball (Player)",
}

item_type_to_text = {
    0: "None",
    1: "Torch",
    2: "Ladder Down",
    3: "Ladder Up",
    4: "Ladder Down Blocked",
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
Potion_Items = [
    "Red Potion",
    "Green Potion",
    "Blue Potion",
    "Pink Potion",
    "Cyan Potion",
    "Yellow Potion",
]
Intrinsic_Items = ["Health", "Food", "Drink", "Energy", "Is Sleeping"]
Direction = ["Left", "Right", "Up", "Down"]

helmet_level_dict = {0: "None", 0.5: "Iron", 1: "Diamond"}

chestplate_level_dict = {0: "None", 0.5: "Iron", 1: "Diamond"}

leggings_level_dict = {0: "None", 0.5: "Iron", 1: "Diamond"}

boots_level_dict = {0: "None", 0.5: "Iron", 1: "Diamond"}

helmet_enchantment_dict = {0: "None", 1: "Fire", 2: "Ice"}

chestplate_enchantment_dict = {0: "None", 1: "Fire", 2: "Ice"}

leggings_enchantment_dict = {0: "None", 1: "Fire", 2: "Ice"}

boots_enchantment_dict = {0: "None", 1: "Fire", 2: "Ice"}

special_values_dict = [
    "Light Level (Day/Night Cycle)",
    "Is Sleeping?",
    "Is Resting?",
    "Learned Fireball?",
    "Learned Iceball?",
    "Current Floor",
    "Is Current Floor Down Ladder Open?",
    "Is Boss Vulnerable?",
]


def generate_distance_dict(max_range=5):
    """Precompute all possible movement descriptions within a given range."""
    distance_dict = {}
    for v in range(-max_range, max_range + 1):
        for h in range(-max_range, max_range + 1):
            if v == 0 and h == 0:
                continue  # Skip (0,0) since it means no movement

            # Vertical description
            vert_text = (
                f"{abs(v)} steps north"
                if v < 0
                else (f"{v} steps south" if v > 0 else "")
            )

            # Horizontal description
            hor_text = (
                f"{abs(h)} steps west"
                if h < 0
                else (f"{h} steps east" if h > 0 else "")
            )

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
    meta_prompt = "You are an intelligent agent exploring the world of Craftax — a procedurally generated, open-ended survival game. "
    meta_prompt += "It is a 2D tile-based environment with nearby tiles visible to you. The world has multiple floors, creatures, items, and hidden dangers. "
    meta_prompt += "Each floor may contain valuable resources and dangerous enemies. "
    meta_prompt += "Your goal is to survive, gather resources, and explore the world. Also, you should complete achievements and progress to get rewards. "
    meta_prompt += "You will receive an observation below describing your current situation, you should focus on what is important to accomplish your goal. "
    meta_prompt += "The first part of the observation contains the information about the blocks within the agent sight. "
    meta_prompt += "The second part contains the information about the items types in your field of vision. "
    meta_prompt += "The next part describes the mobile entities in the world. "
    meta_prompt += "Then you will see the information about your inventory including potions which is useful to craft items. "
    meta_prompt += "The next part contains the information about your intrinsic values. It is important to keep these values high to ensure you survive in the world. "
    meta_prompt += "The last part contains the information about your equipment and their levels along with some special values. "

    text_description.append(meta_prompt)
    text_description.append("Observation: ")

    rows = np.arange(OBS_DIM[0])[:, None]
    cols = np.arange(OBS_DIM[1])[None, :]
    distance_matrix = np.abs(rows - 4) + np.abs(cols - 5)
    max_distance = 100  # max distance to get rid of zeros

    symbolic_array_map = symbolic_array[:8217]
    symbolic_array_map_reshaped = symbolic_array_map.reshape(OBS_DIM[0], OBS_DIM[1], -1)

    # Block types description
    symbolic_array_map_blocks = symbolic_array_map_reshaped[:, :, :37]
    block_types = np.argmax(symbolic_array_map_blocks, axis=-1)
    unique_blocks = np.unique(block_types)
    for block in unique_blocks:
        curr_block_mask = block_types == block
        curr_block_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
        curr_blocks = distance_matrix * curr_block_mask
        curr_blocks_max_dist = np.where(curr_block_mask, curr_blocks, max_distance)
        min_distance_curr_block = np.min(curr_blocks_max_dist)
        min_dist_indices = np.argwhere(curr_blocks_max_dist == min_distance_curr_block)
        relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
        distance_tuples = [tuple(map(int, d)) for d in relative_pos]
        descriptions = [
            distance_lookup.get(d, "Unknown movement") for d in distance_tuples
        ]
        text_description.append(
            Block_id_to_text[block] + " is at: " + ", ".join(descriptions)
        )

    # Item types description
    symbolic_array_map_item = symbolic_array_map_reshaped[:, :, 37:42]
    item_types = np.argmax(symbolic_array_map_item, axis=-1)
    unique_items = np.unique(item_types)
    for item in unique_items:
        curr_item_mask = item_types == item
        curr_item_mask[OBS_DIM[0] // 2, OBS_DIM[1] // 2] = False
        curr_items = distance_matrix * curr_item_mask
        curr_items_max_dist = np.where(curr_item_mask, curr_items, max_distance)
        min_distance_curr_item = np.min(curr_items_max_dist)
        min_dist_indices = np.argwhere(curr_items_max_dist == min_distance_curr_item)
        relative_pos = min_dist_indices - np.array([OBS_DIM[0] // 2, OBS_DIM[1] // 2])
        distance_tuples = [tuple(map(int, d)) for d in relative_pos]
        descriptions = [
            distance_lookup.get(d, "Unknown movement") for d in distance_tuples
        ]
        text_description.append(
            item_type_to_text[item] + " is at: " + ", ".join(descriptions)
        )

    # Mob types description
    symbolic_array_map_mobs = symbolic_array_map_reshaped[:, :, 42:82]
    mob_types = np.argmax(symbolic_array_map_mobs, axis=-1)
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
        descriptions = [
            distance_lookup.get(d, "Unknown movement") for d in distance_tuples
        ]
        text_description.append(
            mob_id_to_text[mob] + " is at: " + ", ".join(descriptions)
        )

    inventory_array = np.argwhere(symbolic_array[8217:8233] > 0).flatten()
    for inv_idx in inventory_array:
        item_count = symbolic_array[8217:8233][inv_idx]
        text_description.append(Inventory_Items[inv_idx] + ": " + str(item_count))

    potions_array = np.argwhere(symbolic_array[8233:8239] > 0).flatten()
    for potion_idx in potions_array:
        potion_count = symbolic_array[8233:8239][potion_idx]
        text_description.append(Potion_Items[potion_idx] + ": " + str(potion_count))

    intrinsic_array = symbolic_array[8239:8248]
    for intrinsic_idx in range(len(Intrinsic_Items)):
        intrinsic_value = intrinsic_array[intrinsic_idx]
        text_description.append(
            Intrinsic_Items[intrinsic_idx] + ": " + str(intrinsic_value * 10)
        )

    direction_array = symbolic_array[8248:8252]
    for direction_idx in range(len(Direction)):
        direction_value = direction_array[direction_idx]
        text_description.append(Direction[direction_idx] + ": " + str(direction_value))

    helmet_level = symbolic_array[8252]
    text_description.append("Helmet Level: " + helmet_level_dict[helmet_level])

    chestplate_level = symbolic_array[8253]
    text_description.append(
        "Chestplate Level: " + chestplate_level_dict[chestplate_level]
    )

    leggings_level = symbolic_array[8254]
    text_description.append("Leggings Level: " + leggings_level_dict[leggings_level])

    boots_level = symbolic_array[8255]
    text_description.append("Boots Level: " + boots_level_dict[boots_level])

    helmet_enchantment = symbolic_array[8256]
    text_description.append(
        "Helmet Enchantment: " + helmet_enchantment_dict[helmet_enchantment]
    )

    chestplate_enchantment = symbolic_array[8257]
    text_description.append(
        "Chestplate Enchantment: " + chestplate_enchantment_dict[chestplate_enchantment]
    )

    leggings_enchantment = symbolic_array[8258]
    text_description.append(
        "Leggings Enchantment: " + leggings_enchantment_dict[leggings_enchantment]
    )

    boots_enchantment = symbolic_array[8259]
    text_description.append(
        "Boots Enchantment: " + boots_enchantment_dict[boots_enchantment]
    )

    special_values = symbolic_array[8260:8268]
    for special_idx in range(len(special_values)):
        special_value = special_values[special_idx]
        text_description.append(
            special_values_dict[special_idx] + ": " + str(special_value)
        )

    return text_description
