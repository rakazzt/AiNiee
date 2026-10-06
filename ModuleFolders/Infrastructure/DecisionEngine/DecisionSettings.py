"""What the consistency sweep is allowed to do.

These are the knobs a user actually wants, separated from the mechanics. They are plain data
with no Qt and no config object, so the sweep can be driven from a test as easily as from the
task, and every field is validated on the way in - the config file is user-editable and may
have been written by an older version.
"""

# Config keys. The selection mirrors how prompts are stored: one record for what is chosen,
# one list for what the user created.
SELECTION_KEY = "extract_decision_settings_selection"
USER_KEY = "extract_decision_settings_user"

DEFAULT_ID = "default"

# The default expresses the intended behaviour end to end: judge every containment pair, keep
# derived names apart, check that a shared name is translated consistently, and refuse glossary
# keys too generic to be safe.
DEFAULT_SETTINGS = {
    "relation_switch": True,        # ask same-entity vs derived-name for each pair
    "consistency_switch": True,     # check a derived name renders the shared name the same way
    "generic_switch": True,         # ask whether a term is too generic to be a safe key
    "drop_single_character": True,  # remove one-character keys (no request, decided by length)
    "drop_generic": True,           # remove terms judged generic; off means report only
    "threshold": 0.5,               # probability cut applied to every answer
    "max_pairs": 400,               # ceiling on pair questions, a cost guard
}

BOOL_KEYS = ("relation_switch", "consistency_switch", "generic_switch",
             "drop_single_character", "drop_generic")

# Inclusive bounds, so a hand-edited config cannot make the sweep act on a coin flip or
# spend an unbounded number of requests.
THRESHOLD_RANGE = (0.05, 0.95)
MAX_PAIRS_RANGE = (1, 5000)


def default_settings() -> dict:
    return dict(DEFAULT_SETTINGS)


def _as_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
    return fallback


def _as_number(value, fallback: float, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(str(value).strip())
        except (TypeError, ValueError):
            return fallback
    value = float(value)
    if value != value:  # NaN
        return fallback
    return min(high, max(low, value))


def normalize(settings) -> dict:
    """Coerce anything into a complete, valid settings map.

    Unknown keys are dropped rather than passed through: these values decide what gets removed
    from the glossary, so an unrecognised key must never look like a policy.
    """
    source = settings if isinstance(settings, dict) else {}
    resolved = {
        key: _as_bool(source.get(key), DEFAULT_SETTINGS[key]) for key in BOOL_KEYS
    }
    resolved["threshold"] = _as_number(
        source.get("threshold"), DEFAULT_SETTINGS["threshold"], *THRESHOLD_RANGE
    )
    resolved["max_pairs"] = int(_as_number(
        source.get("max_pairs"), DEFAULT_SETTINGS["max_pairs"], *MAX_PAIRS_RANGE
    ))
    return resolved


def get_user_settings(config) -> list:
    """Custom settings the user created, in card order, each {id, name, settings}."""
    if not isinstance(config, dict):
        return []
    entries = config.get(USER_KEY)
    if not isinstance(entries, list):
        return []
    cleaned = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        name = entry.get("name")
        if not isinstance(entry_id, str) or not entry_id.strip():
            continue
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned.append({
            "id": entry_id,
            "name": name,
            "settings": normalize(entry.get("settings")),
        })
    return cleaned


def get_selected(config) -> dict:
    """The settings in force. Falls back to the built-in default, never to something partial.

    A selection that points at a deleted custom entry also falls back, so removing the entry a
    user had chosen cannot silently leave the sweep running on stale values.
    """
    if not isinstance(config, dict):
        return default_settings()
    selection = config.get(SELECTION_KEY)
    if not isinstance(selection, dict):
        return default_settings()
    selected_id = selection.get("last_selected_id")
    if selected_id == DEFAULT_ID:
        return default_settings()
    for entry in get_user_settings(config):
        if entry["id"] == selected_id:
            return dict(entry["settings"])
    if isinstance(selected_id, str) and selected_id:
        return default_settings()
    # No id recorded: trust the stored settings, but only as far as they validate.
    return normalize(selection.get("settings"))

# UI labels, kept next to the keys so a new field cannot be added without a name to show.
FIELD_LABELS = {
    "relation_switch": "判定条目的关系（同一实体 / 派生名）",
    "consistency_switch": "检查派生词的公共名称译法是否一致",
    "generic_switch": "判定条目是否过于笼统",
    "drop_single_character": "剔除单字条目（按长度，不消耗请求）",
    "drop_generic": "剔除判定为笼统的条目（关：只报告不剔除）",
    "threshold": "判定阈值（概率低于此值视为无法判定）",
    "max_pairs": "单次巡检最多判定的条目对数量",
}


# The card preview. A composed string still has to be translated as a whole, otherwise an
# English UI shows a Chinese sentence wherever a settings card appears.
SUMMARY_FORMAT = "关系判定 {0}｜一致性检查 {1}｜笼统词判定 {2}｜剔除单字 {3}｜剔除笼统词 {4}｜阈值 {5:.2f}｜上限 {6} 对"


def summarize(settings, translate=None) -> str:
    """One line describing a settings map, for a card preview and for the log.

    `translate` is the caller's translator; without it the Chinese source text is returned,
    which is what the log wants.
    """
    resolved = normalize(settings)
    tra = translate or (lambda text: text)
    on_off = lambda flag: tra("开") if flag else tra("关")
    return tra(SUMMARY_FORMAT).format(
        on_off(resolved["relation_switch"]),
        on_off(resolved["consistency_switch"]),
        on_off(resolved["generic_switch"]),
        on_off(resolved["drop_single_character"]),
        on_off(resolved["drop_generic"]),
        resolved["threshold"],
        resolved["max_pairs"],
    )


def upsert_user_entry(entries, entry) -> list:
    """Insert or replace a custom entry, keeping its position. Returns a new list.

    Replacing in place rather than removing and appending matters: cards are laid out in list
    order, so appending an edit would make the edited card jump to the end.
    """
    result, replaced = [], False
    for existing in entries or []:
        if isinstance(existing, dict) and existing.get("id") == entry["id"]:
            result.append(dict(entry))
            replaced = True
        else:
            result.append(existing)
    if not replaced:
        result.append(dict(entry))
    return result
