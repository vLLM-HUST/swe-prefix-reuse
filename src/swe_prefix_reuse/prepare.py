"""Compile append-only source histories into independently tokenized input deltas.

No truncation, synthetic padding, tool execution, or generated-text retokenization.
Unsupported history-rewriting templates fail explicitly.
"""

import copy
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = "swe-prefix-reuse/v1"


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def compile_session(row, tokenizer):
    messages = copy.deepcopy(row["messages"])
    for message in messages:
        if message.get("content") is not None and not isinstance(message["content"], str):
            raise ValueError("only text trajectories are supported")
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    tools = [json.loads(x) if isinstance(x, str) else x for x in (row.get("tools") or [])]

    def render(history, generation):
        return tokenizer.apply_chat_template(
            history,
            tools=tools,
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=True,
        )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    turns, previous_full, previous_tail, length = [], "", "", 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = render(messages[:index], True)
        full = render(messages[: index + 1], False)
        if not full.startswith(prompt):
            raise ValueError("assistant serialization does not extend generation prefix")
        if not prompt.startswith(previous_full):
            raise ValueError("template rewrites historical messages; cannot preserve prefix")
        # Derive the structural closing delimiter from this template, not a Qwen constant.
        marker = "SWE_PREFIX_REUSE_OUTPUT_SLOT_70c794e3"
        skeleton = render(messages[:index] + [{"role": "assistant", "content": marker}], False)
        if skeleton.count(marker) != 1:
            raise ValueError("cannot locate assistant closing delimiter")
        tail = skeleton.split(marker)[1]
        if not tail or not full.endswith(tail):
            raise ValueError("unsupported assistant closing delimiter")
        body = full[len(prompt) : -len(tail)]
        budget = len(encode(body))
        if budget < 1:
            raise ValueError("empty assistant output budget")
        delta = encode(previous_tail + prompt[len(previous_full) :])
        length += len(delta)
        turns.append(
            {
                "input_ids": delta,
                "output_tokens": budget,
                "prompt_tokens": length,
                "source_message_index": index,
            }
        )
        length += budget
        previous_full, previous_tail = full, tail
    if len(turns) < 2:
        raise ValueError("need at least two assistant turns")
    return {
        "trajectory_id": row["trajectory_id"],
        "instance_id": row["instance_id"],
        "turns": turns,
        "max_context_tokens": length,
    }


def validate_workload(workload):
    if workload.get("schema") != SCHEMA or not workload.get("sessions"):
        raise ValueError("unsupported or empty prepared workload")
    identities = set()
    for session in workload["sessions"]:
        identity = session["trajectory_id"]
        if identity in identities:
            raise ValueError("duplicate trajectory identity")
        identities.add(identity)
        total = 0
        if len(session["turns"]) < 2:
            raise ValueError("session must have multiple turns")
        for turn in session["turns"]:
            ids, budget = turn["input_ids"], turn["output_tokens"]
            if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError("input_ids must be nonnegative integers")
            if type(budget) is not int or budget < 1:
                raise ValueError("output budget must be positive")
            total += len(ids)
            if not total or turn["prompt_tokens"] != total:
                raise ValueError("invalid cumulative prompt shape")
            total += budget
        if session["max_context_tokens"] != total or total > workload["max_context"]:
            raise ValueError("session exceeds context or has inconsistent shape")


def prepare(source, tokenizer_path, output, max_context):
    import transformers
    from jinja2 import TemplateError
    from transformers import AutoTokenizer

    if Path(output).exists():
        raise FileExistsError(output)
    raw = read_json(source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    sessions, rejected = [], []
    for row in raw["sessions"]:
        try:
            session = compile_session(row, tokenizer)
            if session["max_context_tokens"] > max_context:
                raise ValueError("whole trajectory exceeds requested context")
            sessions.append(session)
        except (ValueError, TypeError, KeyError, TemplateError) as exc:
            rejected.append({"trajectory_id": row["trajectory_id"], "reason": str(exc)})
    if not sessions:
        raise ValueError("No accepted sessions: " + json.dumps(rejected))
    # Fingerprint the actual tokenizer backend and template, not its local directory name.
    fingerprint = hashlib.sha256(
        (tokenizer.backend_tokenizer.to_str() + tokenizer.chat_template).encode()
    ).hexdigest()
    result = {
        "schema": SCHEMA,
        "source": raw["provenance"],
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "tokenizer": {
            "path": str(tokenizer_path),
            "fingerprint": fingerprint,
            "transformers": transformers.__version__,
            "enable_thinking": True,
        },
        "max_context": max_context,
        "policy": "independent input-delta and assistant-body tokenization; structural closings are input",
        "sessions": sessions,
        "rejected": rejected,
    }
    validate_workload(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"))
    return {
        "sessions": len(sessions),
        "rejected": rejected,
        "turns": sum(len(s["turns"]) for s in sessions),
        "context_tokens": [s["max_context_tokens"] for s in sessions],
    }
