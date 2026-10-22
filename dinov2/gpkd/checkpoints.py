"""Keep exported evaluation checkpoints compatible with unchunked hub models."""

import re


def unchunk_state_dict(state_dict):
    result = {}
    for key, value in state_dict.items():
        key = re.sub(r"(^|\.)blocks\.\d+\.(\d+)\.", r"\1blocks.\2.", key)
        if key in result:
            raise ValueError(f"Duplicate exported checkpoint key: {key}")
        result[key] = value
    return result
