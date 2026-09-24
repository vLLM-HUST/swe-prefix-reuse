import copy

import pytest

from swe_prefix_reuse.prepare import SCHEMA, compile_session, read_json, validate_workload


class Tokenizer:
    chat_template = "test"

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>" + (m.get("content") or "") + "</end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return list(text.encode())


def source():
    return {
        "trajectory_id": "t",
        "instance_id": "i",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "abc"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "defg"},
        ],
    }


def test_deltas_keep_structural_closing_out_of_output_budget():
    session = compile_session(source(), Tokenizer())
    first, second = session["turns"]
    assert first["output_tokens"] == 3
    assert second["output_tokens"] == 4
    assert bytes(second["input_ids"]).decode() == "</end><tool>observation</end><assistant>"
    history = first["input_ids"] + [12345] * 3 + second["input_ids"]
    assert len(history) == second["prompt_tokens"]
    assert history[len(first["input_ids"]) : len(first["input_ids"]) + 3] == [12345] * 3
    validate_workload({"schema": SCHEMA, "sessions": [session], "max_context": 1000})


def test_history_rewriting_template_rejected():
    class Rewriting(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return text.replace("abc", "rewritten") if len(messages) > 2 else text

    with pytest.raises(ValueError, match="rewrites"):
        compile_session(source(), Rewriting())


def test_invalid_shape_and_context_rejected():
    session = compile_session(source(), Tokenizer())
    workload = {"schema": SCHEMA, "sessions": [session], "max_context": 1}
    with pytest.raises(ValueError, match="context"):
        validate_workload(workload)
    workload["max_context"] = 1000
    bad = copy.deepcopy(workload)
    bad["sessions"][0]["turns"][1]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="shape"):
        validate_workload(bad)


def test_bundled_source_is_small_attributed_and_complete():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/open-swe-sample.json.gz"
    raw = read_json(path)
    assert path.stat().st_size < 1024 * 1024
    assert len(raw["sessions"]) == 8
    assert raw["provenance"]["license"] == "CC-BY-4.0"
    assert len({s["trajectory_id"] for s in raw["sessions"]}) == 8
    assert all(sum(m["role"] == "assistant" for m in s["messages"]) > 1 for s in raw["sessions"])
