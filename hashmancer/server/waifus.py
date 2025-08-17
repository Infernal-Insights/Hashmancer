import random

WAIFU_NAMES = [
    "Asuna",
    "Mikasa",
    "Hinata",
    "Saber",
    "Rem",
    "ZeroTwo",
    "Kurisu",
    "Toga",
    "Rukia",
    "Yoruichi",
    "Akeno",
    "Rias",
    "Mai",
    "Nezuko",
    "Tohru",
    "Kagome",
    "Faye",
    "Motoko",
    "Misato",
    "Homura",
    "Megumin",
    "Nico",
    "Chika",
    "Emilia",
    "Kallen",
    "C.C.",
    "Echidna",
    "Yuno",
    "Albedo",
    "Shalltear",
    "Narberal",
    "Solution",
    "Entoma",
    "Lupusregina",
    "Aura",
    "Mare",
]


def assign_waifu(existing_names):
    unused = list(set(WAIFU_NAMES) - set(existing_names))
    if not unused:
        return f"Waifu-{random.randint(1000, 9999)}"
    return random.choice(unused)
