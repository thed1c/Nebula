"""nebula/schema.py: a small JSON Schema check for typed destinations (the Switchyard validates a decision service's
answer before the caller gets it). The common keywords: type, enum, const, required, properties,
additionalProperties (false), items, minItems/maxItems, minimum/maximum, minLength/maxLength."""
from __future__ import annotations

TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(v, t: str) -> bool:
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, TYPES.get(t, object))


def errors(value, schema: dict, path: str = "$") -> list[str]:
    out: list[str] = []
    if not isinstance(schema, dict):
        return out
    t = schema.get("type")
    if t is not None:
        kinds = t if isinstance(t, list) else [t]
        if not any(_is(value, k) for k in kinds):
            return [f"{path} should be {' or '.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path} should be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        out.append(f"{path} should be {schema['const']!r}")
    if isinstance(value, dict):
        for k in schema.get("required") or []:
            if k not in value:
                out.append(f"{path}.{k} is missing")
        props = schema.get("properties") or {}
        for k, v in value.items():
            if k in props:
                out += errors(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                out.append(f"{path}.{k} is not expected")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{path} has fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            out.append(f"{path} has more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, v in enumerate(value):
                out += errors(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(f"{path} is shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            out.append(f"{path} is longer than {schema['maxLength']}")
    if _is(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            out.append(f"{path} is below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            out.append(f"{path} is above {schema['maximum']}")
    return out
