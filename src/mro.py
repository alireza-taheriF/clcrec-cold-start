"""Synthetic MRO catalog: plants buy parts with planted family structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


INDUSTRIES = {
    "automotive": "قطعه‌سازی و خودرو",
    "steel": "فولاد و نورد",
    "petrochemical": "نفت و پتروشیمی",
    "food": "صنایع غذایی",
    "cement": "سیمان و معدنی",
    "power": "نیروگاه و انرژی",
}

PLANTS = {
    "automotive": [
        "قطعه‌سازی آریا موتور", "ریخته‌گری سپهر خودرو", "تولید قطعات توس‌درایو",
        "خط مونتاژ البرز خودرو", "قالب‌سازی نیوادیزاین",
    ],
    "steel": [
        "مجتمع فولاد زاگرس", "نورد گرم پارس‌آهن", "ذوب القایی کویر",
        "ورق‌سازی اروند", "فولاد آلیاژی بیستون",
    ],
    "petrochemical": [
        "پتروشیمی خلیج‌فام", "پالایش میعانات هرمز", "واحد الفین دشتستان",
        "پلیمر ساحل‌گستر", "گاز مایع کنگان‌شیمی",
    ],
    "food": [
        "لبنیات سپیدرود", "روغن‌کشی دشت طلایی", "کنسرو شمال‌چین",
        "آرد و گلوتن آریادانه", "بسته‌بندی شهدآور",
    ],
    "cement": [
        "سیمان البرز جنوبی", "آهک هیدراته کرمان‌کوه", "گچ ساختمانی کویرپارس",
        "خطوط خردایش بیستون", "معدن سنگ آهن درونه",
    ],
    "power": [
        "نیروگاه سیکل ترکیبی مهرگان", "پست فشار قوی افق", "توربین بخار کارون",
        "توزیع برق ناحیه صنعتی هشتگرد", "واحد CHP پارس‌انرژی",
    ],
}

# category -> industries that actually consume it
CATEGORY_INDUSTRY = {
    "bearing": ["automotive", "steel", "cement"],
    "motor": ["steel", "cement", "food", "power"],
    "valve": ["petrochemical", "power", "food"],
    "sensor": ["automotive", "petrochemical", "food"],
    "plc": ["petrochemical", "power", "steel"],
    "gearbox": ["cement", "steel", "food"],
    "seal": ["petrochemical", "food", "power"],
    "coupling": ["steel", "cement", "power"],
}

CATEGORY_FA = {
    "bearing": "بلبرینگ",
    "motor": "الکتروموتور",
    "valve": "شیر صنعتی",
    "sensor": "سنسور",
    "plc": "ماژول PLC",
    "gearbox": "گیربکس",
    "seal": "مکانیکال سیل",
    "coupling": "کوپلینگ",
}

FAMILIES = [
    ("bearing", "6205-2RS", "ISO", ["SKF", "FAG", "NSK", "Koyo"], "NTN"),
    ("bearing", "6308-C3", "ISO", ["SKF", "FAG", "NSK"], "Timken"),
    ("bearing", "22218-E", "ISO", ["SKF", "FAG"], "NSK"),
    ("bearing", "NU210", "ISO", ["SKF", "FAG", "NTN"], "Koyo"),
    ("motor", "1.5kW-IE2-B3", "IEC", ["Siemens", "ABB", "WEG"], "Motovario"),
    ("motor", "7.5kW-IE3-B5", "IEC", ["Siemens", "ABB"], "Brook"),
    ("motor", "22kW-4P-B3", "IEC", ["ABB", "WEG", "Siemens"], "Leroy"),
    ("motor", "0.75kW-90S", "IEC", ["WEG", "Siemens"], "ABB"),
    ("valve", "DN50-PN16-ball", "EN", ["KSB", "ARI", "Samson"], "Flowserve"),
    ("valve", "DN80-PN40-globe", "EN", ["KSB", "Samson"], "Velan"),
    ("valve", "DN25-PN16-check", "EN", ["ARI", "KSB"], "Samson"),
    ("valve", "1in-300-gate", "ASME", ["Flowserve", "Velan"], "KSB"),
    ("sensor", "PT100-6mm", "IEC", ["Endress", "Wika", "Siemens"], "Omron"),
    ("sensor", "prox-M18-PNP", "IEC", ["Omron", "Sick", "Pepperl"], "Balluff"),
    ("sensor", "4-20mA-pressure", "IEC", ["Wika", "Endress"], "Danfoss"),
    ("sensor", "photo-BGS-100", "IEC", ["Sick", "Omron"], "Pepperl"),
    ("plc", "DI16-24V", "IEC", ["Siemens", "Schneider"], "AllenBradley"),
    ("plc", "DO8-relay", "IEC", ["Siemens", "Omron"], "Schneider"),
    ("plc", "AI4-RTD", "IEC", ["Siemens", "AllenBradley"], "Schneider"),
    ("plc", "CPU-compact", "IEC", ["Schneider", "Siemens"], "Omron"),
    ("gearbox", "helical-i20", "IEC", ["SEW", "Flender", "Nord"], "Bonfiglioli"),
    ("gearbox", "worm-i40", "IEC", ["SEW", "Nord"], "Motovario"),
    ("gearbox", "bevel-i15", "IEC", ["Flender", "SEW"], "Nord"),
    ("seal", "type21-25mm", "ISO", ["JohnCrane", "EagleBurgmann"], "Chesterton"),
    ("seal", "cartridge-45mm", "ISO", ["EagleBurgmann", "JohnCrane"], "AESSEAL"),
    ("seal", "bellow-32mm", "ISO", ["Chesterton", "JohnCrane"], "EagleBurgmann"),
    ("coupling", "jaw-92", "ISO", ["Lovejoy", "KTR"], "Rexnord"),
    ("coupling", "grid-1060", "ISO", ["Falk", "Rexnord"], "KTR"),
    ("coupling", "disc-170", "ISO", ["KTR", "Rexnord"], "Lovejoy"),
]


@dataclass
class MROWorld:
    items: pd.DataFrame
    users: pd.DataFrame
    events: pd.DataFrame
    truth: pd.DataFrame


def generate_mro(seed: int = 7, plants_per_industry: int = 8) -> MROWorld:
    rng = np.random.default_rng(seed)
    users_rows, items_rows, events_rows, truth_rows = [], [], [], []

    user_id = 1
    plants = []
    for ind, names in PLANTS.items():
        extra = max(0, plants_per_industry - len(names))
        all_names = list(names) + [f"{INDUSTRIES[ind]} — واحد {i+1}" for i in range(extra)]
        for name in all_names[:plants_per_industry]:
            plants.append((user_id, name, ind))
            users_rows.append({
                "user_id": user_id,
                "name": name,
                "segment": ind,
                "segment_fa": INDUSTRIES[ind],
            })
            user_id += 1

    item_id = 1000
    families = []
    for category, size, standard, warm_brands, cold_brand in FAMILIES:
        family_id = f"{category}:{size}"
        industries = CATEGORY_INDUSTRY[category]
        warm_ids = []
        for brand in warm_brands:
            title = f"{CATEGORY_FA[category]} {size} {brand}"
            tags = "|".join([category, brand, size, standard, *industries])
            items_rows.append({
                "item_id": item_id,
                "sku": f"{brand[:3].upper()}-{size}",
                "title": title,
                "tags": tags,
                "category": category,
                "category_fa": CATEGORY_FA[category],
                "brand": brand,
                "size": size,
                "standard": standard,
                "family": family_id,
                "is_cold": 0,
            })
            warm_ids.append(item_id)
            item_id += 1

        cold_title = f"{CATEGORY_FA[category]} {size} {cold_brand} (جانشین)"
        cold_tags = "|".join([category, cold_brand, size, standard, *industries, "successor"])
        items_rows.append({
            "item_id": item_id,
            "sku": f"{cold_brand[:3].upper()}-{size}-N",
            "title": cold_title,
            "tags": cold_tags,
            "category": category,
            "category_fa": CATEGORY_FA[category],
            "brand": cold_brand,
            "size": size,
            "standard": standard,
            "family": family_id,
            "is_cold": 1,
        })
        families.append({
            "family": family_id,
            "category": category,
            "industries": industries,
            "warm_ids": warm_ids,
            "cold_id": item_id,
        })
        item_id += 1

    plant_by_ind = {}
    for uid, name, ind in plants:
        plant_by_ind.setdefault(ind, []).append(uid)

    for fam in families:
        industry_plants = []
        for ind in fam["industries"]:
            industry_plants.extend(plant_by_ind[ind])
        n_spec = max(3, len(industry_plants) // 2)
        specialists = set(int(x) for x in rng.choice(industry_plants, size=n_spec, replace=False))
        # specialists buy this exact size/family; others in the industry buy the category via other families
        for uid in specialists:
            n_buy = int(rng.integers(2, min(5, len(fam["warm_ids"]) + 1)))
            chosen = rng.choice(fam["warm_ids"], size=min(n_buy, len(fam["warm_ids"])), replace=False)
            for iid in chosen:
                if rng.random() < 0.94:
                    events_rows.append({"user_id": uid, "item_id": int(iid), "rating": 5})
        for uid in specialists:
            if rng.random() < 0.88:
                truth_rows.append({"user_id": uid, "item_id": fam["cold_id"], "rating": 5})

        # a little cross-traffic so SVD is not a cartoon
        other_users = [uid for uid, _, ind in plants if ind not in fam["industries"]]
        if other_users:
            noise_n = max(1, len(other_users) // 8)
            for uid in rng.choice(other_users, size=noise_n, replace=False):
                iid = int(rng.choice(fam["warm_ids"]))
                events_rows.append({"user_id": int(uid), "item_id": iid, "rating": 5})

    return MROWorld(
        items=pd.DataFrame(items_rows),
        users=pd.DataFrame(users_rows),
        events=pd.DataFrame(events_rows).drop_duplicates(["user_id", "item_id"]),
        truth=pd.DataFrame(truth_rows).drop_duplicates(["user_id", "item_id"]),
    )


def to_movies_frame(items: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "movie_id": items["item_id"],
        "title": items["title"],
        "genres": items["tags"],
    })


def to_ratings_frame(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["movie_id"] = out["item_id"]
    out["timestamp"] = 0
    return out[["user_id", "movie_id", "rating", "timestamp"]]
