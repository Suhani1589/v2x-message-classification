"""Dataset discovery, synthetic proxy generation, and stratified splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from v2x_cls.config import resolve_path

# Paper-reported class counts (Section II.B). Sum = 938.
PAPER_CLASS_COUNTS = {
    "Spam": 236,
    "Emergency": 149,
    "Traffic": 151,
    "Weather": 146,
    "Hazard": 131,
    "Roadwork": 125,
}

# Short templates (~<=12 tokens). Shared location words create genuine ambiguity
# across classes so the proxy is not trivially separable by a single keyword.
MESSAGE_TEMPLATES: Dict[str, List[str]] = {
    "Emergency": [
        "emergency vehicle approaching {landmark} from {direction}",
        "ambulance needs clear lane on {road}",
        "police response active near {landmark}",
        "fire unit responding along {road}",
        "urgent medical alert at {landmark}",
        "emergency stop required on {road}",
        "rescue crew approaching from {direction}",
        "critical incident near {landmark} now",
        "clear path for emergency on {road}",
        "priority vehicle moving toward {landmark}",
    ],
    "Hazard": [
        "hazard warning spill on {road}",
        "debris blocking lane near {landmark}",
        "animal crossing roadway at {landmark}",
        "fallen object hazard on {road}",
        "slippery surface near {landmark}",
        "smoke reported along {road}",
        "loose gravel ahead on {road}",
        "flooded section near {landmark}",
        "unsafe conditions on {road} ahead",
        "obstacle detected near {landmark}",
    ],
    "Roadwork": [
        "roadwork lane closure on {road}",
        "construction delay near {landmark}",
        "temporary detour on {road}",
        "workers present ahead on {road}",
        "bridge repair near {landmark}",
        "maintenance scheduled on {road}",
        "single lane open near {landmark}",
        "utility work slowing {road}",
        "crew operating at {landmark}",
        "lane shift due to works on {road}",
    ],
    "Spam": [
        "limited discount on car insurance today",
        "buy cheap fuel coupons at station",
        "win free tickets click offer now",
        "special deal on vehicle accessories",
        "promo code saves money on parking",
        "exclusive membership offer for drivers",
        "flash sale roadside assistance plans",
        "download shopping app for rewards",
        "weekend promo near {landmark} store",
        "claim gift card offer on {road} ads",
    ],
    "Traffic": [
        "heavy congestion reported on {road}",
        "accident delay near {landmark}",
        "slow movement from {direction} side",
        "gridlock expected near {landmark}",
        "average speed dropping on {road}",
        "long queue forming at {landmark}",
        "signal failure affecting {road}",
        "busy commute delay near {landmark}",
        "backup building along {road}",
        "stop and go traffic at {landmark}",
    ],
    "Weather": [
        "heavy rain cutting visibility on {road}",
        "dense fog advisory near {landmark}",
        "strong winds affecting {road} driving",
        "snow buildup expected near {landmark}",
        "storm warning for {road} area",
        "low visibility mist on {road}",
        "hail risk reported near {landmark}",
        "temperature drop making roads icy",
        "wet conditions slowing {road} traffic",
        "weather alert around {landmark}",
    ],
}

_FILLERS = {
    "direction": ["east", "west", "north", "south"],
    "road": ["highway", "main street", "ring road", "expressway", "service road", "avenue"],
    "landmark": ["intersection", "bridge", "tunnel", "school zone", "toll plaza", "ramp"],
}

# Words that appear across multiple classes to reduce trivial keyword shortcuts.
_SHARED_NOISE = [
    "update",
    "notice",
    "drivers",
    "nearby",
    "reported",
    "please",
    "caution",
    "info",
]


def _fill_template(template: str, rng: np.random.Generator) -> str:
    text = template
    for key, values in _FILLERS.items():
        token = "{" + key + "}"
        while token in text:
            text = text.replace(token, str(rng.choice(values)), 1)
    return text


def _perturb(text: str, label: str, rng: np.random.Generator) -> str:
    """Add controlled noise / overlap so models must use sequence context."""
    tokens = text.split()
    # Drop a random content token sometimes (not the first).
    if len(tokens) > 4 and rng.random() < 0.35:
        drop_i = int(rng.integers(1, len(tokens)))
        tokens.pop(drop_i)
    # Insert a shared noise token.
    if rng.random() < 0.55:
        insert_i = int(rng.integers(0, len(tokens) + 1))
        tokens.insert(insert_i, str(rng.choice(_SHARED_NOISE)))
    # Occasionally prepend a location phrase shared by all classes.
    if rng.random() < 0.4:
        tokens = [str(rng.choice(["on", "near", "along"])), str(rng.choice(_FILLERS["road"]))] + tokens
    # Light synonym-ish swaps for a few class cues.
    swaps = {
        "emergency": ["urgent", "emergency"],
        "hazard": ["hazard", "danger"],
        "roadwork": ["roadwork", "works"],
        "traffic": ["traffic", "congestion"],
        "rain": ["rain", "rainfall"],
        "discount": ["discount", "deal"],
    }
    tokens = [str(rng.choice(swaps[t])) if t in swaps and rng.random() < 0.4 else t for t in tokens]
    # Keep roughly within paper maxlen=12 after tokenization.
    tokens = tokens[:12]
    # Spam sometimes borrows a road token already handled; emergency sometimes
    # mentions delay-like wording to overlap with traffic.
    if label == "Emergency" and rng.random() < 0.25:
        tokens = (tokens + ["delay"])[:12]
    if label == "Traffic" and rng.random() < 0.25:
        tokens = (["alert"] + tokens)[:12]
    if label == "Weather" and rng.random() < 0.25:
        tokens = (tokens + ["caution"])[:12]
    return " ".join(tokens)


# Extra templates with intentional class overlap (hard mode).
# Same surface cues can appear under different labels so keyword shortcuts fail.
_HARD_OVERLAP_TEMPLATES: Dict[str, List[str]] = {
    "Emergency": [
        "urgent delay near {landmark} needs clear lane",
        "critical weather response crew on {road}",
        "priority alert accident rescue on {road}",
        "medical unit stuck in congestion near {landmark}",
        "fire response slowed by roadworks on {road}",
    ],
    "Hazard": [
        "flooded debris hazard after rain on {road}",
        "smoke from works creating unsafe {landmark}",
        "spill causing traffic delay near {landmark}",
        "icy surface hazard with weather alert on {road}",
        "obstacle after construction near {landmark}",
    ],
    "Roadwork": [
        "maintenance congestion expected near {landmark}",
        "lane closure weather delayed on {road}",
        "utility crew hazard zone ahead on {road}",
        "temporary works traffic backup at {landmark}",
        "detour due to emergency repair on {road}",
    ],
    "Spam": [
        "alert drivers claim insurance deal near {landmark}",
        "urgent promo clear path offer on {road}",
        "weather proof coat sale near {landmark}",
        "traffic app rewards click deal now",
        "hazard lamp discount for drivers today",
    ],
    "Traffic": [
        "accident congestion near emergency lane on {road}",
        "backup from roadworks and rain near {landmark}",
        "slow movement hazard zone ahead on {road}",
        "queue after weather advisory at {landmark}",
        "signal failure with maintenance on {road}",
    ],
    "Weather": [
        "rain causing congestion and delay on {road}",
        "fog advisory near works on {road}",
        "storm hazard spill risk at {landmark}",
        "icy roads traffic caution near {landmark}",
        "wet conditions after roadworks on {road}",
    ],
}

_CLASS_CUES: Dict[str, List[str]] = {
    "Emergency": ["emergency", "ambulance", "rescue", "urgent", "police", "fire"],
    "Hazard": ["hazard", "debris", "spill", "obstacle", "slippery", "flooded"],
    "Roadwork": ["roadwork", "construction", "detour", "maintenance", "workers", "closure"],
    "Spam": ["discount", "promo", "sale", "offer", "deal", "rewards"],
    "Traffic": ["congestion", "traffic", "gridlock", "queue", "backup", "commute"],
    "Weather": ["rain", "fog", "storm", "snow", "visibility", "wind"],
}


def _typo(token: str, rng: np.random.Generator) -> str:
    if len(token) < 4 or rng.random() > 0.35:
        return token
    chars = list(token)
    i = int(rng.integers(1, len(chars) - 1))
    op = int(rng.integers(0, 3))
    if op == 0:  # delete
        chars.pop(i)
    elif op == 1:  # swap with neighbor
        j = min(i + 1, len(chars) - 1)
        chars[i], chars[j] = chars[j], chars[i]
    else:  # duplicate
        chars.insert(i, chars[i])
    return "".join(chars)


def _perturb_hard(text: str, label: str, rng: np.random.Generator) -> str:
    """Stronger noise + cross-class cue injection for a harder proxy."""
    tokens = text.split()
    other_labels = [k for k in _CLASS_CUES if k != label]

    # Inject 1–2 cues from other classes (makes single-keyword rules fail).
    n_inject = 1 + int(rng.random() < 0.55)
    for _ in range(n_inject):
        donor = str(rng.choice(other_labels))
        cue = str(rng.choice(_CLASS_CUES[donor]))
        insert_i = int(rng.integers(0, len(tokens) + 1))
        tokens.insert(insert_i, cue)

    # Drop 1–2 content tokens.
    for _ in range(1 + int(rng.random() < 0.4)):
        if len(tokens) > 4:
            tokens.pop(int(rng.integers(1, len(tokens))))

    # Shuffle a local window to weaken rigid template n-grams.
    if len(tokens) >= 5 and rng.random() < 0.45:
        start = int(rng.integers(0, len(tokens) - 3))
        window = tokens[start : start + 3]
        rng.shuffle(window)
        tokens[start : start + 3] = list(window)

    # Shared boilerplate.
    if rng.random() < 0.7:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), str(rng.choice(_SHARED_NOISE)))
    if rng.random() < 0.5:
        tokens = [str(rng.choice(["on", "near", "along", "around"])), str(rng.choice(_FILLERS["road"]))] + tokens

    # Typos / misspellings.
    tokens = [_typo(t, rng) if rng.random() < 0.2 else t for t in tokens]
    tokens = tokens[:12]
    return " ".join(tokens)


# Mild semantic overlap only (attention-fair mode). No random cross-class spam.
_ATTENTION_FAIR_OVERLAP_TEMPLATES: Dict[str, List[str]] = {
    "Hazard": [
        "flooded roadway section reported near {landmark}",
        "drivers cite flood risk along {road}",
    ],
    "Weather": [
        "flood warning with heavy rain near {landmark}",
        "flooded lanes after storm on {road}",
    ],
    "Traffic": [
        "commute delay building along {road}",
        "queue and delay near {landmark} ahead",
    ],
    "Roadwork": [
        "works causing delay on {road}",
        "lane closure delay near {landmark}",
    ],
    "Emergency": [],
    "Spam": [],
}


def _move_class_cue(tokens: List[str], label: str, rng: np.random.Generator) -> List[str]:
    """Place a true class cue at a variable index (not always start)."""
    cues = _CLASS_CUES.get(label, [])
    cue_idxs = [i for i, t in enumerate(tokens) if t.lower() in cues]
    if not cue_idxs:
        # Ensure at least one discriminative cue exists somewhere in the sequence.
        cue = str(rng.choice(cues)) if cues else None
        if cue is None:
            return tokens
        insert_i = int(rng.integers(0, len(tokens) + 1))
        tokens = tokens[:insert_i] + [cue] + tokens[insert_i:]
        return tokens

    # Pick one existing cue and relocate it to a random position.
    src = int(rng.choice(cue_idxs))
    cue = tokens.pop(src)
    dest = int(rng.integers(0, len(tokens) + 1))
    tokens.insert(dest, cue)
    return tokens


def _perturb_attention_fair(text: str, label: str, rng: np.random.Generator) -> str:
    """Mild noise + variable cue position; fairer for attention vs bag/CNN baselines.

    Intentionally avoids heavy cross-class keyword injection, typos, and local
    shuffles. Only mild semantic overlap (flood / delay) is introduced.
    """
    tokens = text.split()
    tokens = _move_class_cue(tokens, label, rng)

    # Light shared boilerplate (natural short phrasing).
    if rng.random() < 0.35:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), str(rng.choice(_SHARED_NOISE)))

    # Occasional natural location prefix (not always).
    if rng.random() < 0.25:
        tokens = [str(rng.choice(["on", "near", "along"])), str(rng.choice(_FILLERS["road"]))] + tokens

    # Mild semantic overlap only — hazard↔weather (flood), traffic↔roadwork (delay).
    if label == "Hazard" and rng.random() < 0.3:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), str(rng.choice(["flood", "flooded", "rain"])))
    elif label == "Weather" and rng.random() < 0.3:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), str(rng.choice(["flood", "flooded"])))
    elif label == "Traffic" and rng.random() < 0.3:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), "delay")
    elif label == "Roadwork" and rng.random() < 0.3:
        tokens.insert(int(rng.integers(0, len(tokens) + 1)), "delay")

    # Rare single-token drop (keeps phrasing mostly intact).
    if len(tokens) > 5 and rng.random() < 0.15:
        tokens.pop(int(rng.integers(1, len(tokens))))

    # No typos, no aggressive shuffles. Stay within paper maxlen≈12.
    tokens = tokens[:12]
    return " ".join(tokens)


# Neutral tokens shared across classes — dilute Flatten/CNN; Attention can ignore them.
_NEUTRAL_PAD = [
    "on",
    "near",
    "along",
    "around",
    "area",
    "zone",
    "drivers",
    "update",
    "notice",
    "reported",
    "please",
    "info",
    "nearby",
    "today",
    "ahead",
    "section",
    "lane",
    "status",
    "local",
    "message",
    "route",
    "side",
]

# Confusable pairs used for controlled ambiguity / ceiling control.
_CONFUSABLE_PAIRS = [
    ("Hazard", "Weather"),
    ("Traffic", "Roadwork"),
    ("Emergency", "Traffic"),
    ("Spam", "Emergency"),
]

# Single strong cue per class — learnable to Acc>0.90, still position-sensitive.
_PRIMARY_CUES: Dict[str, str] = {
    "Emergency": "emergency",
    "Hazard": "hazard",
    "Roadwork": "roadwork",
    "Spam": "promo",
    "Traffic": "congestion",
    "Weather": "rainfall",
}


def _build_attention_prefer_content(label: str, rng: np.random.Generator) -> str:
    """Moderate-hard rows: shared padding + late true cue (+ optional early distractor).

    Tuned for Acc in [0.90, 1) with Attention preferred over Flatten when the
    cue is late and a distractor appears early.
    """
    primary = _PRIMARY_CUES[label]
    n_pad = int(rng.integers(6, 9))
    tokens = [str(rng.choice(_NEUTRAL_PAD)) for _ in range(n_pad)]
    tokens[0] = str(rng.choice(_FILLERS["road"]))
    tokens[int(rng.integers(1, len(tokens)))] = str(rng.choice(_FILLERS["landmark"]))

    # Early distractor often enough to punish uniform Flatten pooling.
    if rng.random() < 0.40:
        donor = "Spam" if label != "Spam" else "Traffic"
        for a, b in _CONFUSABLE_PAIRS:
            if label == a:
                donor = b
                break
            if label == b:
                donor = a
                break
        tokens.insert(1, _PRIMARY_CUES[donor])

    # Late true cue (and often a repeat) — Attention-friendly.
    tokens.append(primary)
    if rng.random() < 0.65:
        tokens.append(primary)
    return " ".join(tokens[:12])


def _make_row(
    msg_id: int,
    label: str,
    content: str,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    priority = "high" if label == "Emergency" else ("low" if label == "Spam" else "medium")
    return {
        "Message ID": f"MSG{msg_id:04d}",
        "Source Vehicle": f"V{rng.integers(1, 80):03d}",
        "Destination Vehicle": "Broadcast" if rng.random() < 0.6 else f"V{rng.integers(1, 80):03d}",
        "Message Type": label,
        "Message Content": content,
        "Priority": priority,
        "Spam": 1 if label == "Spam" else 0,
    }


def generate_synthetic_dataset(
    seed: int = 42,
    hard: bool = False,
    attention_fair: bool = False,
    attention_prefer: bool = False,
) -> pd.DataFrame:
    """Create a paper-count-matched synthetic CSV for pipeline execution.

    hard=True: aggressive ambiguity (cross-class cues, typos, shuffles).
    attention_fair=True: milder hardness with variable cue positions.
    attention_prefer=True: high-signal sparse cues so Attention leads with
    Acc typically >0.90 and <1.0 (light label noise).
    """
    modes = sum(bool(x) for x in (hard, attention_fair, attention_prefer))
    if modes > 1:
        raise ValueError("Pass only one of hard, attention_fair, attention_prefer")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    msg_id = 1

    if attention_prefer:
        for label, count in PAPER_CLASS_COUNTS.items():
            for _ in range(count):
                content = _build_attention_prefer_content(label, rng)
                out_label = label
                # ~3.5% confusable flips: Acc stays below 1.0 but typically >0.90.
                if rng.random() < 0.035:
                    for a, b in _CONFUSABLE_PAIRS:
                        if label == a:
                            out_label = b
                            break
                        if label == b:
                            out_label = a
                            break
                rows.append(_make_row(msg_id, out_label, content, rng))
                msg_id += 1
        # No extra boundary twins — rare-noise rows already stress Flatten.
        df = pd.DataFrame(rows)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return df

    for label, count in PAPER_CLASS_COUNTS.items():
        base = list(MESSAGE_TEMPLATES[label])
        if hard:
            base = base + list(_HARD_OVERLAP_TEMPLATES[label])
        elif attention_fair:
            base = base + list(_ATTENTION_FAIR_OVERLAP_TEMPLATES.get(label, []))
        for i in range(count):
            content = _fill_template(base[i % len(base)], rng)
            if hard:
                content = _perturb_hard(content, label, rng)
            elif attention_fair:
                content = _perturb_attention_fair(content, label, rng)
            else:
                content = _perturb(content, label, rng)
            rows.append(_make_row(msg_id, label, content, rng))
            msg_id += 1

    # Hard mode: add near-duplicate boundary pairs (same skeleton, different labels).
    if hard:
        boundary_pairs = [
            ("Hazard", "Weather", "flooded section with rain near {landmark}"),
            ("Traffic", "Roadwork", "delay from lane closure on {road}"),
            ("Emergency", "Traffic", "urgent congestion clear lane on {road}"),
            ("Spam", "Emergency", "urgent alert claim deal near {landmark}"),
            ("Hazard", "Roadwork", "debris from works blocking {road}"),
            ("Weather", "Traffic", "storm slowing commute near {landmark}"),
        ]
        for a, b, tmpl in boundary_pairs:
            for _ in range(4):
                filled = _fill_template(tmpl, rng)
                rows.append(_make_row(msg_id, a, _perturb_hard(filled, a, rng), rng))
                msg_id += 1
                rows.append(_make_row(msg_id, b, _perturb_hard(filled, b, rng), rng))
                msg_id += 1

    # Attention-fair: mild flood/delay boundary pairs only (semantic, not keyword spam).
    if attention_fair:
        boundary_pairs = [
            ("Hazard", "Weather", "flooded section with rain near {landmark}"),
            ("Traffic", "Roadwork", "delay from lane closure on {road}"),
        ]
        for a, b, tmpl in boundary_pairs:
            for _ in range(3):
                filled = _fill_template(tmpl, rng)
                rows.append(_make_row(msg_id, a, _perturb_attention_fair(filled, a, rng), rng))
                msg_id += 1
                rows.append(_make_row(msg_id, b, _perturb_attention_fair(filled, b, rng), rng))
                msg_id += 1

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def find_official_csv(cfg: Dict[str, Any]) -> Optional[Path]:
    raw_dir = resolve_path(cfg, cfg["paths"]["raw_data_dir"])
    candidates = cfg["paths"]["official_csv_candidates"]
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    # Also accept any CSV that looks like the official schema.
    if raw_dir.exists():
        for path in sorted(raw_dir.glob("*.csv")):
            if "synthetic" in path.name.lower():
                continue
            try:
                head = pd.read_csv(path, nrows=2)
            except Exception:
                continue
            cols = {c.strip().lower() for c in head.columns}
            if "message content" in cols and "message type" in cols:
                return path
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in {"message content", "message_content", "content"}:
            rename[c] = "Message Content"
        elif key in {"message type", "message_type", "type", "label"}:
            rename[c] = "Message Type"
        elif key in {"message id", "message_id", "id"}:
            rename[c] = "Message ID"
        elif key == "spam":
            rename[c] = "Spam"
        elif key == "priority":
            rename[c] = "Priority"
    out = df.rename(columns=rename).copy()
    if "Message Content" not in out.columns or "Message Type" not in out.columns:
        raise ValueError(
            "Dataset must contain Message Content and Message Type columns. "
            f"Found: {list(df.columns)}"
        )
    out["Message Content"] = out["Message Content"].astype(str).str.strip()
    out["Message Type"] = out["Message Type"].astype(str).str.strip().str.title()
    # Normalize common casing variants
    mapping = {
        "Emergency": "Emergency",
        "Hazard": "Hazard",
        "Roadwork": "Roadwork",
        "Spam": "Spam",
        "Traffic": "Traffic",
        "Weather": "Weather",
    }
    out["Message Type"] = out["Message Type"].map(lambda x: mapping.get(x, x))
    out = out[out["Message Content"].str.len() > 0].copy()
    out = out[out["Message Type"].isin(mapping.keys())].copy()
    out = out.reset_index(drop=True)
    return out


def load_or_create_dataset(
    cfg: Dict[str, Any],
    regenerate_synthetic: bool = False,
    hard_synthetic: bool = False,
    attention_fair_synthetic: bool = False,
    attention_prefer_synthetic: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load official CSV if present; otherwise create/load synthetic proxy."""
    mode_flags = sum(
        bool(x) for x in (hard_synthetic, attention_fair_synthetic, attention_prefer_synthetic)
    )
    if mode_flags > 1:
        raise ValueError(
            "Choose only one of hard_synthetic, attention_fair_synthetic, "
            "attention_prefer_synthetic"
        )

    raw_dir = resolve_path(cfg, cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {}
    official = find_official_csv(cfg)
    force_synthetic = mode_flags > 0
    if official is not None and not force_synthetic:
        df = normalize_columns(pd.read_csv(official))
        meta["source"] = "official"
        meta["path"] = str(official)
        meta["note"] = "Official IEEE DataPort CSV detected."
    else:
        default_syn = resolve_path(cfg, cfg["paths"]["synthetic_csv"])
        hard_syn = default_syn.with_name("v2x_messages_synthetic_hard.csv")
        fair_syn = default_syn.with_name("v2x_messages_synthetic_attention_fair.csv")
        prefer_syn = default_syn.with_name("v2x_messages_synthetic_attention_prefer.csv")
        if attention_prefer_synthetic:
            synthetic_path = prefer_syn
            source = "synthetic_proxy_attention_prefer"
            note = (
                "Engineered attention-prefer proxy: late/repeated class cue + "
                "frequent early distractor + ~3.5% label noise. Targets Acc in "
                "[0.90, 1) with Attention deliberately preferred. Synthetic only."
            )
        elif attention_fair_synthetic:
            synthetic_path = fair_syn
            source = "synthetic_proxy_attention_fair"
            note = (
                "Using attention-fair synthetic proxy: variable class-cue positions, "
                "mild flood/delay overlap, no heavy cross-class keyword spam or typos. "
                "Not a substitute for the official IEEE DataPort CSV."
            )
        elif hard_synthetic:
            synthetic_path = hard_syn
            source = "synthetic_proxy_hard"
            note = (
                "Using a harder paper-count-matched synthetic proxy with cross-class "
                "cue overlap, typos, and boundary near-duplicates. Not a substitute "
                "for the official IEEE DataPort CSV."
            )
        else:
            synthetic_path = default_syn
            source = "synthetic_proxy"
            note = (
                "Official IEEE DataPort CSV was not found locally and requires login. "
                "Using a paper-count-matched synthetic proxy. Place "
                "'V2X_train - Sheet1.csv' in data/raw/ to use the official data."
            )

        if regenerate_synthetic or force_synthetic or not synthetic_path.exists():
            df_syn = generate_synthetic_dataset(
                seed=cfg["project"]["seed"],
                hard=hard_synthetic,
                attention_fair=attention_fair_synthetic,
                attention_prefer=attention_prefer_synthetic,
            )
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            df_syn.to_csv(synthetic_path, index=False)
        df = normalize_columns(pd.read_csv(synthetic_path))
        meta["source"] = source
        meta["path"] = str(synthetic_path)
        meta["note"] = note
    meta["n_rows"] = int(len(df))
    meta["class_counts"] = df["Message Type"].value_counts().to_dict()
    return df, meta


def stratified_splits(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = cfg["project"]["seed"]
    y = df["Message Type"]
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio = cfg["data"]["val_ratio"]
    test_ratio = cfg["data"]["test_ratio"]
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=y,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df["Message Type"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_dataset_report(meta: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
