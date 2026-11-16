# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prefill-chunk / decode interleaving (VLLM_RBLN_INTERLEAVE_POLICY)."""

import pytest
from vllm.v1.outputs import ModelRunnerOutput

from vllm_rbln.v1.core.interleave_policy import (
    CHUNK,
    DECODE,
    IDLE,
    InterleavePolicy,
)

from .utils import create_requests, create_scheduler

CHUNK_SIZE = 16


# ---------------------------------------------------------------- policy only
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance_ms(self, ms):
        self.t += ms / 1000.0


def test_policy_validation():
    assert not InterleavePolicy("none").enabled
    assert InterleavePolicy("FIXED").enabled
    with pytest.raises(ValueError):
        InterleavePolicy("budget")
    with pytest.raises(ValueError):
        InterleavePolicy("fixed", k=0)
    with pytest.raises(ValueError):
        InterleavePolicy("slack", slack_ms=0)


@pytest.mark.parametrize("mode", ["fixed", "slack"])
def test_policy_trivial_cases(mode):
    p = InterleavePolicy(mode)
    assert p.decide(has_prefill_work=False, num_decodable=0) == IDLE
    assert p.decide(has_prefill_work=False, num_decodable=3) == DECODE
    # nothing to decode: prefill runs chunk after chunk
    for _ in range(5):
        assert p.decide(has_prefill_work=True, num_decodable=0) == CHUNK
        p.on_scheduled(CHUNK)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_policy_fixed_k(k):
    p = InterleavePolicy("fixed", k=k)
    seq = []
    for _ in range(3 * (k + 1)):
        a = p.decide(has_prefill_work=True, num_decodable=2)
        p.on_scheduled(a, num_decode=2 if a == DECODE else 0)
        seq.append(a)
    assert seq == ([CHUNK] * k + [DECODE]) * 3


def test_policy_slack():
    clk = FakeClock()
    p = InterleavePolicy("slack", slack_ms=30.0, clock=clk)

    def step(kind, ms, n=2):
        p.on_scheduled(kind, num_decode=n if kind == DECODE else 0)
        clk.advance_ms(ms)
        p.on_output()

    # Before any measurement: one chunk after a decode, then decode (no estimate).
    assert p.decide(True, 2) == CHUNK
    step(CHUNK, 10.0)
    assert p.decide(True, 2) == DECODE
    step(DECODE, 10.0)
    # After a decode: always a chunk (starvation guard), even if it overshoots.
    assert p.decide(True, 2) == CHUNK
    step(CHUNK, 10.0)
    # now - last decode = 10 ms; +chunk 10 +decode 10 = 30 <= 30 -> another chunk
    assert p.est_chunk() == pytest.approx(0.010)
    assert p.est_decode(2) == pytest.approx(0.010)
    assert p.predicted_gap(2) == pytest.approx(0.030)
    assert p.decide(True, 2) == CHUNK
    step(CHUNK, 10.0)
    # 20 + 10 + 10 = 40 > 30 -> decode
    assert p.decide(True, 2) == DECODE
    step(DECODE, 10.0)
    assert p.decide(True, 2) == CHUNK


def test_policy_slack_estimates_by_batch():
    clk = FakeClock()
    p = InterleavePolicy("slack", clock=clk)
    for n, ms in [(1, 6.0), (4, 8.0), (8, 11.0)]:
        p.on_scheduled(DECODE, num_decode=n)
        clk.advance_ms(ms)
        p.on_output()
    assert p.est_decode(4) == pytest.approx(0.008)
    assert p.est_decode(3) == pytest.approx(0.008)  # next larger seen
    assert p.est_decode(9) == pytest.approx(0.011)  # largest seen


# ---------------------------------------------------------------- scheduler
def _runner_output(scheduler, sched_out) -> ModelRunnerOutput:
    """Fake runner: no token for an intermediate prefill chunk."""
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    sampled = []
    for rid in req_ids:
        r = scheduler.requests[rid]
        sampled.append([] if r.num_computed_tokens < r.num_prompt_tokens else [7])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _kind(scheduler, out):
    """('chunk', rid, n) / ('decode', sorted rids) / ('idle',)."""
    if not out.num_scheduled_tokens:
        return ("idle",)
    items = list(out.num_scheduled_tokens.items())
    if len(items) == 1:
        rid, n = items[0]
        r = scheduler.requests[rid]
        if r.num_computed_tokens - n < r.num_prompt_tokens:
            return ("chunk", rid, n)
    assert all(n == 1 for _, n in items)
    return ("decode", tuple(sorted(rid for rid, _ in items)))


def _make(monkeypatch, policy, max_num_seqs=4, prefix=False, k=1):
    monkeypatch.setenv("VLLM_RBLN_INTERLEAVE_POLICY", policy)
    monkeypatch.setenv("VLLM_RBLN_INTERLEAVE_K", str(k))
    max_model_len = 128
    if prefix:
        ib = CHUNK_SIZE
        return create_scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_model_len,
            num_blocks=max_model_len // ib * max_num_seqs + 1,
            block_size=ib,
            max_model_len=max_model_len,
            outer_block_size=max_model_len,
            enable_prefix_caching=True,
            additional_config={"prefill_chunk_size": CHUNK_SIZE},
        )
    # one KV block per sequence, like the optimum compiled models
    return create_scheduler(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_model_len,
        num_blocks=max_num_seqs + 1,
        block_size=max_model_len,
        max_model_len=max_model_len,
        additional_config={"prefill_chunk_size": CHUNK_SIZE},
    )


def _run(scheduler, steps, add_at=None):
    """Run `steps` engine steps. add_at: {step_index: [requests]}."""
    trace = []
    for i in range(steps):
        for r in (add_at or {}).get(i, []):
            scheduler.add_request(r)
        out = scheduler.schedule()
        trace.append((_kind(scheduler, out), out))
        scheduler.update_from_output(out, _runner_output(scheduler, out))
    return trace


def test_default_policy_is_unchanged(monkeypatch):
    monkeypatch.delenv("VLLM_RBLN_INTERLEAVE_POLICY", raising=False)
    sched = create_scheduler(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        num_blocks=5,
        block_size=128,
        max_model_len=128,
    )
    assert not sched.interleave.enabled
    reqs = create_requests(num_requests=2, num_tokens=64, max_tokens=8)
    sched.add_request(reqs[0])
    out = sched.schedule()
    # whole prompt in one step, no decode alongside
    assert out.num_scheduled_tokens == {"0": 64}
    assert out.prefill_pad_target is None


@pytest.mark.parametrize("prefix", [False, True])
def test_fixed_k1_alternates(monkeypatch, prefix):
    sched = _make(monkeypatch, "fixed", prefix=prefix)
    a, b = create_requests(num_requests=2, num_tokens=CHUNK_SIZE * 4, max_tokens=30)
    a.prompt_token_ids  # noqa: B018
    trace = _run(sched, 14, add_at={0: [a], 6: [b]})
    kinds = [t[0] for t in trace]
    # A: 4 chunks with nothing to decode, then decode only
    assert kinds[:4] == [("chunk", "0", CHUNK_SIZE)] * 4
    assert kinds[4:6] == [("decode", ("0",))] * 2
    # B arrives: chunk / decode(A) alternate, B joins the batch after its last chunk
    assert kinds[6:14] == [
        ("chunk", "1", CHUNK_SIZE),
        ("decode", ("0",)),
        ("chunk", "1", CHUNK_SIZE),
        ("decode", ("0",)),
        ("chunk", "1", CHUNK_SIZE),
        ("decode", ("0",)),
        ("chunk", "1", CHUNK_SIZE),
        ("decode", ("0", "1")),
    ]
    # B's first chunk is a new request, later chunks are cached requests
    assert len(trace[6][1].scheduled_new_reqs) == 1
    assert trace[8][1].scheduled_cached_reqs.req_ids == ["1"]
    assert trace[8][1].scheduled_cached_reqs.num_computed_tokens == [CHUNK_SIZE]
    # decode steps during B's prefill carry B's pad target (block, progress)
    assert trace[7][1].prefill_pad_target is not None
    assert trace[7][1].prefill_pad_target[1] == CHUNK_SIZE
    assert trace[9][1].prefill_pad_target[1] == 2 * CHUNK_SIZE
    assert trace[13][1].prefill_pad_target is None
    # B got exactly one token (from its last chunk) before the joint decode
    assert sched.requests["1"].num_output_tokens == 2


def test_fixed_k2(monkeypatch):
    sched = _make(monkeypatch, "fixed", k=2)
    a, b = create_requests(num_requests=2, num_tokens=CHUNK_SIZE * 4, max_tokens=30)
    trace = _run(sched, 12, add_at={0: [a], 5: [b]})
    kinds = [t[0][0] for t in trace[5:12]]
    assert kinds == ["chunk", "chunk", "decode", "chunk", "chunk", "decode", "decode"]


def test_short_prompt_single_chunk(monkeypatch):
    sched = _make(monkeypatch, "fixed")
    (a,) = create_requests(num_requests=1, num_tokens=10, max_tokens=3)
    trace = _run(sched, 4, add_at={0: [a]})
    assert trace[0][0] == ("chunk", "0", 10)
    assert sched.prefilling is None
    assert [t[0][0] for t in trace[1:3]] == ["decode", "decode"]
    assert sched.requests.get("0") is None  # finished after 3 tokens


def test_one_prefill_at_a_time(monkeypatch):
    sched = _make(monkeypatch, "fixed")
    reqs = create_requests(num_requests=3, num_tokens=CHUNK_SIZE * 2, max_tokens=20)
    trace = _run(sched, 8, add_at={0: reqs})
    kinds = [t[0] for t in trace]
    assert kinds[0] == ("chunk", "0", CHUNK_SIZE)
    assert kinds[1] == ("chunk", "0", CHUNK_SIZE)  # nothing to decode yet
    assert kinds[2] == ("decode", ("0",))
    assert kinds[3] == ("chunk", "1", CHUNK_SIZE)
    assert kinds[4] == ("decode", ("0",))
    assert kinds[5] == ("chunk", "1", CHUNK_SIZE)
    assert kinds[6] == ("decode", ("0", "1"))
    assert kinds[7] == ("chunk", "2", CHUNK_SIZE)


def test_all_blocks_owned_prefix_caching(monkeypatch):
    """3 slots: 2 decoding + 1 prefilling own every outer block. The default
    path would call get_dummy_block() and fail; the interleaved path pads into
    the prefilling block at its progress instead."""
    sched = _make(monkeypatch, "fixed", max_num_seqs=3, prefix=True)
    reqs = create_requests(num_requests=3, num_tokens=CHUNK_SIZE * 3, max_tokens=40)
    trace = _run(sched, 16, add_at={0: reqs})
    decodes_during_prefill = [
        out
        for (kind, out) in trace
        if kind[0] == "decode"
        and out.prefill_pad_target is not None
        and len(out.num_scheduled_tokens) == 2
    ]
    assert decodes_during_prefill
    for out in decodes_during_prefill:
        assert out.dummy_block is None
        blk, pos = out.prefill_pad_target
        assert pos % CHUNK_SIZE == 0 and pos > 0
        assert blk == int(sched.kv_cache_manager.get_block_table("2").flatten()[0])


def test_slack_scheduler_uses_measured_steps(monkeypatch):
    sched = _make(monkeypatch, "slack")
    clk = FakeClock()
    sched.interleave.clock = clk
    a, b = create_requests(num_requests=2, num_tokens=CHUNK_SIZE * 6, max_tokens=60)
    sched.add_request(a)
    kinds = []

    def step(ms):
        out = sched.schedule()
        kinds.append(_kind(sched, out))
        clk.advance_ms(ms)
        sched.update_from_output(out, _runner_output(sched, out))

    for _ in range(8):  # A: 6 chunks, then decodes (2 ms each)
        step(2.0)
    sched.add_request(b)
    for _ in range(10):  # chunk 2 ms, decode 2 ms, X = 30 ms -> several chunks fit
        step(2.0)
    b_part = [k for k in kinds[8:] if k[0] == "chunk"]
    assert len(b_part) == 6
    # with cheap steps Slack runs consecutive chunks (fixed k=1 never would)
    assert ("chunk", "1", CHUNK_SIZE) == kinds[8] == kinds[9]
