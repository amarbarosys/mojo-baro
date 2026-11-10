import json

import jsonschema


def is_valid(schema_text: str, sample_text: str) -> bool:
    schema = json.loads(schema_text)
    try:
        obj = json.loads(sample_text)
    except json.JSONDecodeError:
        return False
    try:
        jsonschema.validate(obj, schema)
        return True
    except jsonschema.exceptions.ValidationError:
        return False


if __name__ == "__main__":
    import sys

    schema_text = open(sys.argv[1]).read()
    sample_text = sys.stdin.read()
    ok = is_valid(schema_text, sample_text)
    print("VALID" if ok else "INVALID")
    sys.exit(0 if ok else 1)
