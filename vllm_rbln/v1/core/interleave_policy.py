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
"""Step-level decision rule for interleaving prefill chunks with decode steps.

On static-graph NPUs the prefill graph and the decode graph are separate, so a
step is either ONE prefill chunk of the (single) prefilling request or ONE
decode step of all running requests. The current optimum path runs every chunk
of a prompt back-to-back, which stalls ongoing decodes for the whole prefill.

This module only decides *which* of the two to run next. It holds no vLLM
objects so it can be unit-tested in isolation.

Policies
  none   : current behaviour (not used by the interleaved scheduler).
  fixed  : run up to ``k`` chunks, then one decode step, repeat.
  slack  : after a decode step always run one chunk (no starvation). After a
           chunk, run another chunk only if the predicted gap of the oldest
           decoding request stays within ``slack_ms``:
              (now - last decode output) + est(next chunk) + est(next decode)
           Step-time estimates are measured online by the scheduler
           (schedule() -> update_from_output()), so server overhead is included.
"""

from __future__ import annotations

import time
from collections.abc import Callable

CHUNK = "chunk"
DECODE = "decode"
IDLE = "idle"

POLICIES = ("none", "fixed", "slack")


class _Ewma:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> None:
        self.value = (
            x
            if self.value is None
            else (self.alpha * x + (1.0 - self.alpha) * self.value)
        )


class InterleavePolicy:
    def __init__(
        self,
        mode: str = "none",
        k: int = 1,
        slack_ms: float = 30.0,
        ewma_alpha: float = 0.5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        mode = mode.lower()
        if mode not in POLICIES:
            raise ValueError(f"Unknown interleave policy {mode!r}; choose {POLICIES}")
        if k < 1:
            raise ValueError("interleave k must be >= 1")
        if slack_ms <= 0:
            raise ValueError("slack_ms must be > 0")
        self.mode = mode
        self.k = k
        self.slack_s = slack_ms / 1000.0
        self.clock = clock
        self._alpha = ewma_alpha
        # decision state
        self.last_kind: str | None = None
        self.consecutive_chunks = 0
        # timing state (seconds, clock())
        self._t_sched: float | None = None
        self._sched_kind: str | None = None
        self._sched_n_decode = 0
        self.t_last_decode_out: float | None = None
        self.t_last_out: float | None = None
        self._chunk_est = _Ewma(ewma_alpha)
        self._decode_est: dict[int, _Ewma] = {}
        # counters (for logging / tests)
        self.num_chunk_steps = 0
        self.num_decode_steps = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    # ------------------------------------------------------------------ estimates
    def est_chunk(self) -> float | None:
        return self._chunk_est.value

    def est_decode(self, n: int) -> float | None:
        if n in self._decode_est:
            return self._decode_est[n].value
        # fall back to the closest larger batch we have seen, then the largest
        larger = sorted(b for b in self._decode_est if b > n)
        if larger:
            return self._decode_est[larger[0]].value
        if self._decode_est:
            return max(e.value for e in self._decode_est.values())
        return None

    def predicted_gap(
        self, n_after_chunk: int, now: float | None = None
    ) -> float | None:
        """Gap (s) the oldest decoding request would see if one more chunk ran now
        and was followed by a decode step of ``n_after_chunk`` requests. None if
        there is not enough measurement yet."""
        now = self.clock() if now is None else now
        ref = (
            self.t_last_decode_out
            if self.t_last_decode_out is not None
            else self.t_last_out
        )
        c = self.est_chunk()
        d = self.est_decode(n_after_chunk)
        if ref is None or c is None or d is None:
            return None
        return (now - ref) + c + d

    # ------------------------------------------------------------------ decision
    def decide(
        self,
        has_prefill_work: bool,
        num_decodable: int,
        chunk_is_last: bool = False,
        now: float | None = None,
    ) -> str:
        """Return CHUNK, DECODE or IDLE.

        has_prefill_work: a prefill is in progress, or a waiting request can start.
        num_decodable: running requests that have finished prefill.
        chunk_is_last: the next chunk completes the prompt (its request then joins
            the decode batch, so the following decode has one more row).
        """
        if not has_prefill_work:
            return DECODE if num_decodable > 0 else IDLE
        if num_decodable == 0:
            return CHUNK
        if self.mode == "fixed":
            return DECODE if self.consecutive_chunks >= self.k else CHUNK
        if self.mode == "slack":
            if self.last_kind != CHUNK:
                return CHUNK  # after a decode (or at start) always progress prefill
            gap = self.predicted_gap(num_decodable + (1 if chunk_is_last else 0), now)
            if gap is not None and gap <= self.slack_s:
                return CHUNK
            return DECODE
        raise RuntimeError(f"decide() called with policy {self.mode!r}")

    # ------------------------------------------------------------------ bookkeeping
    def on_scheduled(
        self, kind: str, num_decode: int = 0, now: float | None = None
    ) -> None:
        """Record the action chosen for this step (called at the end of schedule())."""
        if kind == IDLE:
            self._t_sched = None
            self._sched_kind = None
            return
        self._t_sched = self.clock() if now is None else now
        self._sched_kind = kind
        self._sched_n_decode = num_decode
        self.last_kind = kind
        if kind == CHUNK:
            self.consecutive_chunks += 1
            self.num_chunk_steps += 1
        else:
            self.consecutive_chunks = 0
            self.num_decode_steps += 1

    def on_output(self, now: float | None = None) -> None:
        """Record the end of the step (called from update_from_output())."""
        if self._t_sched is None:
            return
        now = self.clock() if now is None else now
        dt = now - self._t_sched
        if self._sched_kind == CHUNK:
            self._chunk_est.update(dt)
        else:
            n = self._sched_n_decode
            if n not in self._decode_est:
                self._decode_est[n] = _Ewma(self._alpha)
            self._decode_est[n].update(dt)
            self.t_last_decode_out = now
        self.t_last_out = now
        self._t_sched = None
        self._sched_kind = None

    def reset_prefill(self) -> None:
        """Forget chunk streak (e.g. the prefilling request was preempted/aborted)."""
        self.consecutive_chunks = 0
