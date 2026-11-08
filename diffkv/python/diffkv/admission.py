# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-side reservations backed by the worker's measured physical pages."""


class DiffKVAdmission:
    def __init__(self, total_pages: int, num_layers: int, num_kv_heads: int):
        if min(total_pages, num_layers, num_kv_heads) <= 0:
            raise ValueError("DiffKV admission dimensions must be positive")
        self.total_pages = total_pages
        self.num_kv_caches = num_layers * num_kv_heads
        self.free_pages = total_pages
        self.requests: dict[str, dict] = {}
        self.begin_step()

    def begin_step(self):
        self.available_pages = self.free_pages
        self.reservations: dict[str, int] = {}
        self.preempted: set[str] = set()

    @property
    def usage(self) -> float:
        return 1 - self.free_pages / self.total_pages

    def prompt_pages(self, length: int) -> int:
        return ((length + 31) // 32 + 1) * self.num_kv_caches

    def reserve(self, request_id: str, num_computed_tokens: int, length: int) -> bool:
        if request_id in self.reservations:
            return True
        state = self.requests.get(request_id)
        if state is None and num_computed_tokens > 0:
            raise RuntimeError(
                "A running DiffKV request is missing from the worker report"
            )
        is_new = num_computed_tokens == 0 or state is None
        if is_new:
            required = self.prompt_pages(length)
            if required > self.total_pages:
                raise ValueError(
                    f"DiffKV prompt {request_id} requires {required} physical "
                    f"pages, but the arena holds {self.total_pages}"
                )
        elif state is not None and state["phase"] == "prefill":
            required = 0
        else:
            required = self.num_kv_caches * 2
        if required > self.available_pages:
            return False
        self.available_pages -= required
        self.reservations[request_id] = required
        return True

    def cancel(self, request_id: str):
        self.available_pages += self.reservations.pop(request_id, 0)

    def preempt(self, request_id: str):
        """Credit only explicit victims, released before this worker step."""
        self.cancel(request_id)
        if request_id in self.preempted:
            return
        self.preempted.add(request_id)
        state = self.requests.get(request_id)
        if state is not None:
            self.available_pages += state["packed_pages"]

    def update_report(self, report: dict):
        if report is None:
            raise RuntimeError("DiffKV worker did not return a physical cache report")
        if report["total_pages"] != self.total_pages or report["page_bytes"] != 6784:
            raise RuntimeError("DiffKV worker and scheduler arena sizes differ")
        free = report["free_pages"]
        requests = report["requests"]
        allocated = 0
        for state in requests.values():
            packed = state["packed_pages"]
            if type(packed) is not int or packed < 0:
                raise RuntimeError("Invalid DiffKV request page count")
            allocated += packed
            if state["phase"] not in ("prefill", "decode"):
                raise RuntimeError("Invalid DiffKV request phase")
        if (
            type(free) is not int
            or not 0 <= free <= self.total_pages
            or free + allocated != self.total_pages
        ):
            raise RuntimeError("DiffKV physical page accounting is inconsistent")
        self.free_pages = free
        self.requests = requests
