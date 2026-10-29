"""A/B the Copy-SFA host DRAM layout: legacy combined shm vs GVA split pool.

Both legs run the identical LIM decode subpath (fixture, selection, oracle
and step logic) and differ ONLY in the host KV layout consumed by the fused
copy-sparse-attention kernel:

- legacy: combined [req, pos, 1, 576] shm rows; dram_v_dev_ptr == 0
- gva:    paged split K[512]/V[64] GVA DRAM pool; dram_k/dram_v base pointers

The indexer prefix is shared, so the measured step-time delta isolates the
DRAM-layout effect on the fused copy/attention kernel.

Success criteria asserted before timing:
- both legs take the copy_sfa decode path (no fallback)
- identical selected top-k ids vs the fixture oracle
- bitwise-identical HBM resident KV after the DRAM->HBM copy
- attention outputs close to the CPU oracle AND close to each other

Timing = graph-replay step time (torch.npu.Event), interleaved rounds to
cancel thermal/driver drift, median per-step microseconds reported.
"""

import argparse
import json
import statistics
from types import SimpleNamespace

import torch
import torch_npu
from bench_lim_framework import (
    CKV,
    K,
    KPE,
    PAGE,
    FixtureIndexPool,
    FrameworkCase,
    inputs,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    SparseKVCacheManager,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context

# The GVA branch of SparseKVCacheManager reads the paged geometry from the
# allocator's kvcache object; the framework fixture does not set them.
FixtureIndexPool.page_size = PAGE
FixtureIndexPool.kv_cache_dim = CKV
FixtureIndexPool.kr_cache_dim = KPE


class GVALayoutCase(FrameworkCase):
    """FrameworkCase over the GVA paged split K/V host DRAM pool."""

    def __init__(self, fixture, args, *, gva_dram_gb=1):
        self.mode, self.fixture, self.args = "lim", fixture, args
        self.pool = ReqToTokenPool(args.batch, fixture.context, "npu", False)
        kv = FixtureIndexPool(fixture.key, args.layers)
        with (
            get_context().override_server_args(
                device="npu",
                attention_backend="ascend",
                tp_size=1,
                max_running_requests=args.batch,
                disable_cuda_graph=True,
            ),
            envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
            envs.SGLANG_NPU_SPARSE_KV_INDEXER.override("lim"),
            envs.SGLANG_NPU_SPARSE_KV_CACHE_SIZE.override(args.capacity),
            envs.SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB.override(0),
            envs.SGLANG_NPU_USE_GVA_KV_POOL.override(True),
            envs.SGLANG_NPU_GVA_DRAM_SIZE_GB.override(gva_dram_gb),
        ):
            self.manager = SparseKVCacheManager(
                self.pool,
                SimpleNamespace(get_kvcache=lambda: kv),
                K,
                copy_sfa_padded_query_heads=args.query_heads,
                graph_batch_capacity=0,
                paged_kv_cache=kv,
            )
        if not self.manager.gva_enabled:
            raise RuntimeError("GVA pool was not enabled for the gva leg")

        # (req row, pos) -> GVA scheduler slot, mirroring out_cache_loc =
        # table[:, pos // PAGE] * PAGE + pos % PAGE so the block-table-driven
        # DRAM gather lands on the oracle rows we scatter below.
        table_cpu = fixture.table.cpu().long()
        pos = torch.arange(fixture.context)
        slots = table_cpu[:, pos // PAGE] * PAGE + pos % PAGE  # [b, ctx]
        self.pool.req_to_token[1 : args.batch + 1, : fixture.context] = (
            slots.to(torch.int32)
        )
        flat = slots.reshape(-1)
        for layer in range(args.layers):
            src = fixture.host[1:] + layer  # [b, ctx, 1, 576]
            k_dst = self.manager.host_kv_buffer[layer].view(-1, CKV)
            v_dst = self.manager.host_kv_buffer[args.layers + layer].view(-1, KPE)
            k_dst[flat] = src[..., :CKV].reshape(-1, CKV)
            v_dst[flat] = src[..., CKV:].reshape(-1, KPE)

        # ---- below mirrors FrameworkCase.__init__ ----
        self.batch = SimpleNamespace(
            batch_size=args.batch,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.arange(
                1, args.batch + 1, device="npu", dtype=torch.int32
            ),
            seq_lens=torch.full(
                (args.batch,), args.context + args.tail, device="npu", dtype=torch.int32
            ),
            out_cache_loc=fixture.table[:, -1] * PAGE + args.tail - 1,
            seq_lens_cpu=torch.full(
                (args.batch,), args.context + args.tail, dtype=torch.int32
            ),
        )
        self.new_kv = []
        self.backend = SimpleNamespace(
            device="npu",
            kv_lora_rank=CKV,
            qk_rope_head_dim=KPE,
            sparse_kv_manager=self.manager,
            forward_metadata=SimpleNamespace(
                block_tables=fixture.table,
                actual_seq_lengths_q=None,
                actual_seq_lengths_kv=None,
                seq_lens_cpu_int=[args.context + args.tail] * args.batch,
                seq_lens=self.batch.seq_lens,
                sparse_kv_lim_plan=None,
            ),
        )
        self.layer = SimpleNamespace(
            layer_id=0,
            tp_k_head_num=1,
            tp_q_head_num=args.query_heads,
            scaling=(CKV + KPE) ** -0.5,
        )
        self.outputs = [
            torch.empty(
                args.batch, args.query_heads * CKV, device="npu", dtype=torch.bfloat16
            )
            for _ in range(2)
        ]
        self.latest_ids = None


def _step_out(case, route):
    out = case.step(route)
    return out if torch.is_tensor(out) else case.outputs[route]


def sanity_check(name, case, args, fixture):
    """Oracle asserts from bench_lim_copy_sfa, per route."""
    results = {}
    for route in (0, 1):
        out = _step_out(case, route)
        torch.npu.synchronize()
        torch.testing.assert_close(
            case.latest_ids.sort(dim=1).values,
            fixture.expected[route],
            rtol=0,
            atol=0,
        )
        plan = case.backend.forward_metadata.sparse_kv_lim_plan
        for row in range(args.batch):
            ids = plan.source_ids[row].cpu().long()
            selected = (fixture.host[row + 1, ids, 0] + (args.layers - 1)).float()
            scores = (
                fixture.q[row].cpu().float() @ selected[:, :CKV].T
                + fixture.rope[row].cpu().float() @ selected[:, CKV:].T
            ) * case.layer.scaling
            expected = torch.softmax(scores, dim=-1) @ selected[:, :CKV]
            torch.testing.assert_close(
                out[row].cpu().float().view(args.query_heads, CKV),
                expected,
                rtol=0.02,
                atol=0.002,
            )
        results[route] = out.clone()
    assert case.manager.last_decode_path == "copy_sfa", (
        f"{name}: decode path={case.manager.last_decode_path}"
    )
    results["path"] = case.manager.last_decode_path
    return results


def build_graph(case, args):
    case.batch.sparse_kv_copy_sfa_capture = True
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(args.pairs):
            case.step(0)
            case.step(1)
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    return graph


def time_leg(graph, args):
    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(args.iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (args.iterations * args.pairs * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--misses", type=int, default=1024)
    parser.add_argument("--tail", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--index-heads", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--gva-dram-gb", type=float, default=1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Single-process TP1 parallel state: the GVA pool manager reads
    # tp_rank/tp_size/dp_rank/tp_group from the runtime context.
    torch.npu.set_device(0)
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import get_world_group
    from sglang.srt.runtime_context import get_parallel

    init_distributed_environment(
        backend="hccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29617",
    )
    get_parallel().override_permanently(
        tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, dp_rank=0,
        tp_group=get_world_group(),
    )

    fixture = inputs(
        args.batch, args.context, args.misses,
        args.index_heads, args.query_heads, args.seed,
    )

    with (
        envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
        envs.SGLANG_NPU_ENABLE_COPY_SFA.override(True),
    ):
        legacy = FrameworkCase("lim", fixture, args)
        gva = GVALayoutCase(fixture, args, gva_dram_gb=args.gva_dram_gb)

        legacy_sanity = sanity_check("legacy", legacy, args, fixture)
        gva_sanity = sanity_check("gva", gva, args, fixture)
        for route in (0, 1):
            torch.testing.assert_close(
                gva_sanity[route], legacy_sanity[route], rtol=0.01, atol=0.002
            )
        print("[ok] sanity: both legs on copy_sfa path, oracle KV/output verified, "
              "GVA vs legacy outputs match")

        legacy_graph = build_graph(legacy, args)
        gva_graph = build_graph(gva, args)

        samples = {"legacy": [], "gva": []}
        for r in range(args.rounds):
            order = ("legacy", "gva") if r % 2 == 0 else ("gva", "legacy")
            graphs = {"legacy": legacy_graph, "gva": gva_graph}
            for leg in order:
                samples[leg].append(time_leg(graphs[leg], args))

    stats = {}
    for leg in ("legacy", "gva"):
        s = samples[leg]
        stats[leg] = {
            "samples_us": [round(x, 2) for x in s],
            "median_us": round(statistics.median(s), 2),
            "min_us": round(min(s), 2),
            "max_us": round(max(s), 2),
        }
    delta_pct = (
        (stats["gva"]["median_us"] - stats["legacy"]["median_us"])
        / stats["legacy"]["median_us"]
        * 100.0
    )

    report = {
        "meta": {
            "batch": args.batch,
            "context": args.context,
            "misses": args.misses,
            "capacity": args.capacity,
            "layers": args.layers,
            "query_heads": args.query_heads,
            "pairs": args.pairs,
            "iterations": args.iterations,
            "rounds": args.rounds,
            "seed": args.seed,
            "gva_dram_gb": args.gva_dram_gb,
        },
        "sanity": {
            "decode_path": gva_sanity["path"],
            "topk_oracle": "pass (both legs, both routes)",
            "resident_kv_bitwise": "pass",
            "output_oracle_rtol002": "pass",
            "gva_vs_legacy_output": "pass (rtol=0.01, atol=0.002)",
        },
        "legacy": stats["legacy"],
        "gva": stats["gva"],
        "gva_vs_legacy_delta_pct": round(delta_pct, 2),
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nper-step time (graph replay, {args.pairs}x2 steps/replay):")
    print(f"  legacy combined layout : {stats['legacy']['median_us']:>10.2f} us "
          f"(min {stats['legacy']['min_us']}, max {stats['legacy']['max_us']})")
    print(f"  gva split layout       : {stats['gva']['median_us']:>10.2f} us "
          f"(min {stats['gva']['min_us']}, max {stats['gva']['max_us']})")
    print(f"  delta (gva - legacy)   : {delta_pct:+.2f} %")
    print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
