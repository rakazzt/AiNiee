"""Describe what a settings save actually changed.

The config file is the app's whole behaviour surface: which interface each role uses,
which model and request-shaping fields it sends, which prompts are selected. A user turned
think_switch off and set temperature to 1.00, watched an extraction that had failed for a
day start succeeding - and the log held no record of either change. Reconstructing it took
hours of diffing logs by hand.

save_config already reads the previous file and receives the new one on every write, so it
is the single place that can record this. Keeping the diff here - free of Qt, of the config
object and of the logger - means the rule is testable without a running app.

Two things this must never do: leak a key, or describe a value so long it drowns the log.
"""

# A config migration can touch many keys at once; past this many lines the rest is counted
# rather than listed, so one save cannot become a wall.
MAX_REPORTED_CHANGES = 12

# Values that must never reach the log in readable form.
SECRET_FIELDS = ("api_key", "access_key", "secret_key", "password", "token")

# Only these platform fields change what goes on the wire. Listing the others (names, tags,
# model catalogues) would bury the ones that explain behaviour.
REQUEST_SHAPING_FIELDS = (
    "profile", "api_format", "api_url", "model", "think_switch", "think_depth",
    "thinking_level", "thinking_budget", "temperature", "extra_body", "rpm_limit",
    "auto_complete", "tls_switch",
)

# Roles an interface can be bound to, in the order a person thinks about them.
ROLES = ("active", "extract", "translate", "polish", "proofread")

_VALUE_LIMIT = 48


def _text(value, limit: int = _VALUE_LIMIT) -> str:
    """A value short enough for one log line."""
    if value is None:
        return "(未设置)"
    if isinstance(value, bool):
        return "开" if value else "关"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "(空)"
        return text if len(text) <= limit else text[:limit] + "…"
    if isinstance(value, dict):
        return "{{{} 项}}".format(len(value))
    if isinstance(value, (list, tuple)):
        return "[{} 项]".format(len(value))
    return str(value)[:limit]


def _secret_state(value) -> str:
    """Whether a key is set - never which key."""
    return "(已设置)" if value else "(已清空)"


def _platform_name(platforms, tag) -> str:
    if not isinstance(platforms, dict):
        return str(tag)
    platform = platforms.get(tag)
    if isinstance(platform, dict) and platform.get("name"):
        return str(platform["name"])
    return str(tag)


def describe(old, new, limit: int = MAX_REPORTED_CHANGES) -> list:
    """Human-readable lines for what changed. Empty when nothing did."""
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    lines = []

    old_settings = old.get("api_settings") if isinstance(old.get("api_settings"), dict) else {}
    new_settings = new.get("api_settings") if isinstance(new.get("api_settings"), dict) else {}
    old_platforms = old.get("platforms") if isinstance(old.get("platforms"), dict) else {}
    new_platforms = new.get("platforms") if isinstance(new.get("platforms"), dict) else {}

    # Which interface each role points at. This is the "which provider am I using" record.
    for role in ROLES:
        before, after = old_settings.get(role), new_settings.get(role)
        if before != after:
            lines.append("接口绑定 {0}: {1} → {2}".format(
                role, _platform_name(old_platforms, before), _platform_name(new_platforms, after)))

    for tag in sorted(set(new_platforms) - set(old_platforms)):
        lines.append("新增接口: {0}".format(_platform_name(new_platforms, tag)))
    for tag in sorted(set(old_platforms) - set(new_platforms)):
        lines.append("删除接口: {0}".format(_platform_name(old_platforms, tag)))

    # Request-shaping fields on interfaces that still exist. This is the record of the kind
    # of change that silently caused - or silently fixed - an outage.
    for tag in sorted(set(old_platforms) & set(new_platforms)):
        before, after = old_platforms.get(tag), new_platforms.get(tag)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        name = _platform_name(new_platforms, tag)
        for field in REQUEST_SHAPING_FIELDS:
            if before.get(field) == after.get(field):
                continue
            if field in SECRET_FIELDS:
                lines.append("接口设置 {0}: {1} {2} → {3}".format(
                    name, field, _secret_state(before.get(field)), _secret_state(after.get(field))))
            else:
                lines.append("接口设置 {0}: {1} - {2} → {3}".format(
                    name, field, _text(before.get(field)), _text(after.get(field))))

        # A key swap must be visible without being readable.
        if before.get("api_key") != after.get("api_key"):
            lines.append("接口设置 {0}: api_key {1} → {2}".format(
                name, _secret_state(before.get("api_key")), _secret_state(after.get("api_key"))))

    if len(lines) > limit:
        hidden = len(lines) - limit
        lines = lines[:limit] + ["… 另有 {0} 项改动未列出".format(hidden)]
    return lines
