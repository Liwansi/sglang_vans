"""Opt-in admission snapshots; never synchronize devices or change scheduling."""

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class PrefillBatchDebug:
    def __init__(self, **snapshot):
        self.snapshot = snapshot
        self.skipped = Counter()
        self.stop_reason = "waiting_queue_exhausted"
        self.stop_details = {}

    def skip(self, reason):
        self.skipped[reason] += 1

    def stop(self, reason, **details):
        self.stop_reason = reason
        self.stop_details = details

    def emit(self, *, free_request_slots, adder=None):
        record = dict(self.snapshot)
        record.update(
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
            skipped=dict(self.skipped),
            free_request_slots_after=free_request_slots,
            selected_requests=0,
        )
        if adder is not None:
            record.update(
                selected_requests=len(adder.can_run_list),
                policy_stop=getattr(adder, "admission_debug", {}),
                remaining_input_tokens=adder.rem_input_tokens,
                remaining_chunk_tokens=adder.rem_chunk_tokens,
                remaining_total_tokens=adder.rem_total_tokens,
                remaining_current_tokens=adder.cur_rem_tokens,
                chunk_continues=adder.new_chunked_req is not None,
                selected_extend_tokens=[
                    req.extend_input_len for req in adder.can_run_list
                ],
            )
        logger.info(
            "PREFILL_BATCH_DEBUG %s",
            json.dumps(record, separators=(",", ":"), default=str),
        )
