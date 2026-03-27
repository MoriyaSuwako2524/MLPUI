"""Graph execution engine — topological sort then execute each node."""

from __future__ import annotations
import json
import os
from collections import defaultdict
from datetime import datetime

_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")


def execute_prompt(prompt: dict) -> dict:
    """Execute a serialised node graph.

    prompt format::

        {
            "1": {
                "class_type": "LoadMLPModel",
                "inputs": {"model_name": "uma-s-1p1.pt", "task": "omol"}
            },
            "3": {
                "class_type": "CalculateEnergyForces",
                "inputs": {
                    "calculator": ["1", 0],   # node-1, output index 0
                    "atoms":      ["2", 0]
                }
            }
        }
    """
    from app.nodes import NODE_CLASS_MAPPINGS

    order = _topological_sort(prompt)
    cache: dict[str, list] = {}   # node_id -> list of output values
    ui_results: dict[str, dict] = {}

    for node_id in order:
        node_data = prompt[node_id]
        class_type = node_data["class_type"]

        if class_type not in NODE_CLASS_MAPPINGS:
            raise ValueError(f"Unknown node type: {class_type!r}")

        node_class = NODE_CLASS_MAPPINGS[class_type]
        instance = node_class()

        # Resolve inputs: references → actual values, literals pass through
        kwargs: dict = {}
        for key, val in node_data.get("inputs", {}).items():
            if _is_link(val):
                ref_id, ref_idx = val
                kwargs[key] = cache[str(ref_id)][ref_idx]
            else:
                kwargs[key] = val

        func = getattr(instance, node_class.FUNCTION)
        outputs = func(**kwargs)
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        cache[node_id] = list(outputs)

        if getattr(node_class, "OUTPUT_NODE", False):
            ui_results[node_id] = {
                "class_type": class_type,
                "outputs": _serialise_outputs(outputs, node_class),
            }

    result = {"status": "success", "ui": ui_results}
    _save_run(prompt, result)
    return result


def _save_run(prompt: dict, result: dict) -> None:
    os.makedirs(_output_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
    data = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt":    prompt,
        "result":    result,
    }
    path = os.path.join(_output_dir, f"{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# ── helpers ─────────────────────────────────────────────────────────────────

def _is_link(val) -> bool:
    """Return True if val is a [node_id, output_index] reference."""
    return (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and isinstance(val[1], int)
    )


def _serialise_outputs(outputs, node_class) -> list:
    """Convert Python objects to JSON-safe representations."""
    result = []
    types = list(getattr(node_class, "RETURN_TYPES", []))
    names = list(getattr(node_class, "RETURN_NAMES", types))
    for i, val in enumerate(outputs):
        t = types[i] if i < len(types) else "UNKNOWN"
        n = names[i] if i < len(names) else str(i)
        if isinstance(val, (int, float, str, bool, type(None))):
            result.append({"name": n, "type": t, "value": val})
        else:
            result.append({"name": n, "type": t, "value": repr(val)})
    return result


def _topological_sort(prompt: dict) -> list[str]:
    deps: dict[str, set] = {nid: set() for nid in prompt}

    for node_id, node_data in prompt.items():
        for val in node_data.get("inputs", {}).values():
            if _is_link(val):
                ref_id = str(val[0])
                if ref_id in prompt:
                    deps[node_id].add(ref_id)

    remaining = {nid: set(d) for nid, d in deps.items()}
    order: list[str] = []
    queue = [nid for nid, d in remaining.items() if not d]

    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for other in list(remaining):
            if nid in remaining[other]:
                remaining[other].discard(nid)
                if not remaining[other]:
                    queue.append(other)

    if len(order) != len(prompt):
        raise ValueError("Cycle detected in node graph")

    return order
