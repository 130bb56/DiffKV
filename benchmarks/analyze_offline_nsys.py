#!/usr/bin/env python3
"""Break down one whole-generation Nsight Systems SQLite capture.

No GPU work is executed. Kernel/category summed durations are reported alongside
an additive wall-time partition so concurrent GPU work is never counted twice.
"""

import argparse
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

CATEGORIES = (
    "gemm_including_split_k_reduce", "attention_main", "attention_reduce",
    "attention_other_or_fp16", "kv_cache", "other",
)
WALL_BUCKETS = tuple("kernel:" + name for name in CATEGORIES) + (
    "kernel:overlap", "memcpy_only", "memset_only",
    "memcpy_memset_overlap_only", "gpu_idle",
)
PID_MASK = ~((1 << 24) - 1)


def classify_kernel(name):
    name = name.lower()
    if ("reduce_kernel" in name and "vllm" in name) or "paged_attention_v2_reduce" in name:
        return "attention_reduce"
    if any(s in name for s in ("sparse_attention", "gqa_attention", "sparse_paged_attention")):
        return "attention_main"
    if any(s in name for s in ("attention", "attn_", "flash_fwd")):
        return "attention_other_or_fp16"
    if any(s in name for s in ("reshape_and_cache", "cache", "quantize", "evict", "compact", "prun")):
        return "kv_cache"
    if any(s in name for s in ("gemm", "cutlass", "splitkreduce", "split_k", "gemv")):
        return "gemm_including_split_k_reduce"
    return "other"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def measure(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def wall_partition(events, start, end):
    """Partition the wall interval by active category, prioritizing kernels."""
    times = Counter({name: 0 for name in WALL_BUCKETS})
    category_unions = Counter()
    active = Counter()
    previous = start
    busy_intervals = []
    for timestamp, label, delta in sorted(events):
        if timestamp > previous:
            width = timestamp - previous
            kernel_categories = [key for key in active if key in CATEGORIES]
            for key in kernel_categories:
                category_unions[key] += width
            if kernel_categories:
                bucket = "kernel:" + kernel_categories[0] if len(kernel_categories) == 1 else "kernel:overlap"
            elif "memcpy" in active and "memset" in active:
                bucket = "memcpy_memset_overlap_only"
            elif "memcpy" in active:
                bucket = "memcpy_only"
            elif "memset" in active:
                bucket = "memset_only"
            else:
                bucket = "gpu_idle"
            times[bucket] += width
            if active:
                if busy_intervals and busy_intervals[-1][1] == previous:
                    busy_intervals[-1] = (busy_intervals[-1][0], timestamp)
                else:
                    busy_intervals.append((previous, timestamp))
            previous = timestamp
        active[label] += delta
        if active[label] == 0:
            del active[label]
        if active.get(label, 0) < 0:
            raise ValueError("Unbalanced GPU interval endpoints")
    if active:
        raise ValueError("Unclosed GPU interval endpoints")
    times["gpu_idle"] += end - previous
    if sum(times.values()) != end - start:
        raise ValueError("Wall partition does not conserve time")
    return times, category_unions, busy_intervals


def analyze(sqlite_path, capture_metadata=None, range_name="measured_generation", device=0):
    sqlite_path = Path(sqlite_path).resolve()
    metadata_path = Path(capture_metadata).resolve() if capture_metadata else None
    metadata = json.loads(metadata_path.read_text()) if metadata_path else {}
    connection = sqlite3.connect(sqlite_path.as_uri() + "?mode=ro", uri=True)
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    nvtx = []
    for a, b, text, text_id, tid in connection.execute(
            "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end>start"):
        nvtx.append((a, b, text or strings.get(text_id, ""), tid))
    ranges = [r for r in nvtx if r[2] == range_name]
    if len(ranges) != 1:
        raise ValueError(f"Expected one {range_name!r} NVTX range, found {len(ranges)}")
    start, end, _, owner_tid = ranges[0]
    wall_ns = end - start
    owner_pid = owner_tid & PID_MASK if owner_tid is not None else None
    def ms(value):
        return value / 1_000_000
    def pct(value):
        return value / wall_ns * 100
    def selected(rows):
        return [r for r in rows if r[0] >= start and r[1] <= end]

    step_metadata = {r["nvtx_name"]: r for r in metadata.get("engine_steps", [])}
    outer = selected([r for r in nvtx if re.fullmatch(r"engine_step_\d+", r[2])])
    if outer:
        if not step_metadata:
            raise ValueError("engine_step_N ranges require capture metadata with engine_steps phases")
        scope_source = "outer engine_step_N ranges, phases from capture_metadata.engine_steps"
        scopes = []
        for a, b, name, tid in outer:
            if name not in step_metadata:
                raise ValueError("Missing engine-step metadata: " + name)
            phase = step_metadata[name].get("phase")
            if phase not in ("prefill", "decode", None):
                raise ValueError("Unknown engine-step phase: " + str(phase))
            scopes.append((a, b, phase or "other", tid, name))
    else:
        scope_source = "legacy prefill/decode NVTX ranges; no outer engine_step_N ranges"
        scopes = [(a,b,"prefill" if name == "prefill" else "decode",tid,name)
                  for a,b,name,tid in selected(nvtx)
                  if name == "prefill" or re.fullmatch(r"decode_step_\d+", name)]
    scope_indices = {}
    for tid in {r[3] for r in scopes}:
        ordered = sorted(r for r in scopes if r[3] == tid)
        if any(a[1] > b[0] for a,b in zip(ordered,ordered[1:])):
            raise ValueError("Selected outer engine-step ranges overlap on a CPU thread")
        scope_indices[tid] = ([r[0] for r in ordered], ordered)
    def phase_at(timestamp, tid):
        if tid not in scope_indices:
            return "unattributed"
        starts, entries = scope_indices[tid]
        index = bisect.bisect_right(starts,timestamp)-1
        if index >= 0 and timestamp < entries[index][1]:
            return entries[index][2]
        return "unattributed"

    restrictions = "start < ? AND end > ? AND deviceId = ?"
    parameters = [end,start,device]
    if owner_pid is not None:
        restrictions += " AND globalPid = ?"
        parameters.append(owner_pid)
    fields = ("start,end,demangledName,shortName,correlationId,globalPid,"
              "gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,"
              "staticSharedMemory,dynamicSharedMemory,streamId")
    kernels = connection.execute("SELECT " + fields + " FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE " + restrictions,parameters).fetchall()
    if not kernels:
        raise ValueError("No GPU kernels found in the selected measured-generation range/device/process")
    gpu_memory = []
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY","memcpy"),("CUPTI_ACTIVITY_KIND_MEMSET","memset")):
        if table not in tables:
            continue
        columns = {r[1] for r in connection.execute('PRAGMA table_info("'+table+'")')}
        operation = "copyKind" if "copyKind" in columns else "NULL"
        byte_count = "bytes" if "bytes" in columns else "0"
        query = f"SELECT start,end,correlationId,globalPid,{byte_count},{operation} FROM {table} WHERE {restrictions}"
        gpu_memory.extend((a,b,corr,pid,nbytes,operation_id,kind) for a,b,corr,pid,nbytes,operation_id in connection.execute(query,parameters))
    copy_labels = {}
    if "ENUM_CUDA_MEMCPY_OPER" in tables:
        copy_labels = {r[0]:r[1] for r in connection.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER")}

    api_groups = defaultdict(lambda:[0,0])
    api_intervals, wait_intervals = [], []
    correlation_phase = {}
    gpu_correlations = {r[4] for r in kernels} | {r[2] for r in gpu_memory}
    runtime_tables = [name for name in ("CUPTI_ACTIVITY_KIND_RUNTIME","CUPTI_ACTIVITY_KIND_DRIVER") if name in tables]
    api_kind = {key:(value,"launch" in value.lower(),bool(re.search(r"Synchronize|Wait",value))) for key,value in strings.items()}
    runtime_rows = 0
    for table in runtime_tables:
        where = "start < ? AND end > ?"
        values = [end,start]
        if owner_pid is not None:
            where += " AND (globalTid & ?) = ?"
            values.extend((PID_MASK,owner_pid))
        for a,b,tid,corr,name_id in connection.execute(f"SELECT start,end,globalTid,correlationId,nameId FROM {table} WHERE {where}",values):
            runtime_rows += 1
            name,is_launch,is_wait = api_kind.get(name_id,(str(name_id),False,False))
            left,right = max(a,start),min(b,end)
            row = api_groups[name]
            row[0] += 1
            row[1] += right-left
            if tid == owner_tid:
                api_intervals.append((left,right))
                if is_wait:
                    wait_intervals.append((left,right))
            if corr in gpu_correlations:
                phase = phase_at(a,tid)
                # Prefer a valid containing scope over a nested API that cannot
                # be matched. GPU graph nodes may remain unattributed.
                if phase != "unattributed" or corr not in correlation_phase:
                    correlation_phase[corr] = phase

    name_cache = {key:(value,classify_kernel(value)) for key,value in strings.items()}
    kernel_groups = defaultdict(lambda:[0,0,Counter()])
    phase_kernel_groups = defaultdict(lambda:defaultdict(lambda:[0,0]))
    category_groups = {name:[0,0] for name in CATEGORIES}
    events = []
    clipped_kernels = 0
    kernel_sum = 0
    phase_counts = Counter()
    configuration_keys = ("gridX","gridY","gridZ","blockX","blockY","blockZ",
                          "registersPerThread","staticSharedMemory","dynamicSharedMemory","streamId")
    for row in kernels:
        a,b,name_id,short_id,corr,pid = row[:6]
        left,right = max(a,start),min(b,end)
        elapsed = right-left
        clipped_kernels += int(a<start or b>end)
        name,category = name_cache.get(name_id,name_cache.get(short_id,(str(name_id),"other")))
        group = kernel_groups[name]
        group[0] += 1
        group[1] += elapsed
        group[2][row[6:]] += 1
        category_groups[category][0] += 1
        category_groups[category][1] += elapsed
        kernel_sum += elapsed
        phase = correlation_phase.get(corr,"unattributed")
        phase_counts[phase] += 1
        phase_group = phase_kernel_groups[phase][name]
        phase_group[0] += 1
        phase_group[1] += elapsed
        events.extend(((left,category,1),(right,category,-1)))
    memory_groups = defaultdict(lambda:[0,0,0])
    phase_memory = defaultdict(lambda:[0,0])
    memory_clipped = 0
    for a,b,corr,pid,nbytes,operation_id,kind in gpu_memory:
        left,right = max(a,start),min(b,end)
        memory_clipped += int(a<start or b>end)
        label = copy_labels.get(operation_id,str(operation_id)) if kind == "memcpy" else "memset"
        group = memory_groups[(kind,label)]
        group[0] += 1
        group[1] += right-left
        group[2] += nbytes or 0
        phase = correlation_phase.get(corr,"unattributed")
        phase_memory[phase][0] += 1
        phase_memory[phase][1] += right-left
        events.extend(((left,kind,1),(right,kind,-1)))
    wall_times,category_unions,gpu_busy = wall_partition(events,start,end)
    del events
    api_union,wait_union = merged(api_intervals),merged(wait_intervals)
    gpu_active_ns = measure(gpu_busy)
    api_gpu_ns = intersection(api_union,gpu_busy)
    wait_gpu_ns = intersection(wait_union,gpu_busy)
    kernel_union_ns = sum(value for key,value in wall_times.items() if key.startswith("kernel:"))

    def kernel_rows(groups,include_config=False):
        rows = []
        for name,group in sorted(groups.items(),key=lambda item:item[1][1],reverse=True):
            count,total = group[:2]
            result = {"name":name,"category":classify_kernel(name),"calls":count,
                      "time_ms":ms(total),"whole_wall_pct":pct(total),"mean_us":total/count/1000}
            if include_config:
                result["configurations"] = [{**dict(zip(configuration_keys,config)),"calls":n} for config,n in group[2].most_common()]
            rows.append(result)
        return rows
    phases = {}
    all_scope_intervals = []
    for phase in ("prefill","decode","other","unattributed"):
        intervals = merged([(a,b) for a,b,p,tid,name in scopes if p == phase])
        all_scope_intervals.extend(intervals)
        cpu_ns = measure(intervals)
        groups = phase_kernel_groups.get(phase,{})
        category_totals = {name:[0,0] for name in CATEGORIES}
        for name,(calls,total) in groups.items():
            cat = classify_kernel(name)
            category_totals[cat][0] += calls
            category_totals[cat][1] += total
        total = sum(group[1] for group in groups.values())
        n_scopes = sum(1 for r in scopes if r[2] == phase)
        phases[phase] = {
            "scope_count":n_scopes,"cpu_scope_ms":ms(cpu_ns),"cpu_scope_whole_wall_pct":pct(cpu_ns),
            "mean_cpu_scope_ms":ms(cpu_ns)/n_scopes if n_scopes else None,
            "kernel_calls":sum(group[0] for group in groups.values()),"kernel_sum_ms":ms(total),
            "kernel_whole_wall_pct":pct(total),
            "mean_kernel_ms_per_scope":ms(total)/n_scopes if n_scopes else None,
            "gpu_active_during_cpu_scopes_ms":ms(intersection(intervals,gpu_busy)),
            "no_recorded_gpu_during_cpu_scopes_ms":ms(cpu_ns-intersection(intervals,gpu_busy)),
            "memory_operation_calls":phase_memory[phase][0],"memory_duration_sum_ms":ms(phase_memory[phase][1]),
            "categories":[{"name":name,"calls":calls,"time_ms":ms(duration),"whole_wall_pct":pct(duration)} for name,(calls,duration) in category_totals.items()],
            "top_kernels":kernel_rows(groups),
        }
    scope_union_ns = measure(merged(all_scope_intervals))
    matched_metadata_steps = (len(outer)==len(step_metadata)) if outer else None
    attributed_calls = len(kernels)-phase_counts["unattributed"]
    attributed_time = sum(group[1] for phase,groups in phase_kernel_groups.items() if phase != "unattributed" for group in groups.values())
    kernel_rows_all = kernel_rows(kernel_groups,True)
    warnings = []
    if clipped_kernels or memory_clipped:
        warnings.append("GPU activity crosses the NVTX generation boundary; durations are clipped to that boundary.")
    if phase_counts["unattributed"]:
        warnings.append("Some GPU kernels lack a CUDA API correlation inside a selected engine-step scope; they are retained under unattributed.")
    if not scopes:
        warnings.append("No prefill/decode step scopes were found; phase CPU coverage is unavailable.")
    if outer and not matched_metadata_steps:
        warnings.append("Capture metadata engine-step count differs from captured outer NVTX ranges.")
    result = {
        "schema_version":1,"sqlite_path":str(sqlite_path),"sqlite_sha256":file_sha256(sqlite_path),
        "analysis_script_sha256":file_sha256(__file__),"analyzed_utc":datetime.now(timezone.utc).isoformat(),
        "capture_metadata_path":str(metadata_path) if metadata_path else None,"capture_metadata":metadata,
        "case":metadata.get("case"),"backend":metadata.get("backend"),
        "measurement":{"range_name":range_name,"start_ns":start,"end_ns":end,
                       "device_id":device,"global_pid":owner_pid,"global_tid":owner_tid,
                       "phase_scope_source":scope_source,"units":{"time_ms":"milliseconds","mean_us":"microseconds","whole_wall_pct":"100 * time / measured_generation wall"}},
        "wall_ms":ms(wall_ns),"gpu_busy_ms":ms(gpu_active_ns),"gpu_busy_whole_wall_pct":pct(gpu_active_ns),
        "kernel_sum_ms":ms(kernel_sum),"kernel_union_ms":ms(kernel_union_ns),
        "kernel_concurrency_excess_ms":ms(kernel_sum-kernel_union_ns),
        "wall_breakdown":[{"name":name,"time_ms":ms(wall_times[name]),"whole_wall_pct":pct(wall_times[name])} for name in WALL_BUCKETS],
        "categories":[{"name":name,"calls":calls,"time_ms":ms(total),"whole_wall_pct":pct(total),
                       "union_ms":ms(category_unions[name]),"union_whole_wall_pct":pct(category_unions[name]),
                       "disjoint_wall_ms":ms(wall_times['kernel:'+name]),"disjoint_whole_wall_pct":pct(wall_times['kernel:'+name])} for name,(calls,total) in category_groups.items()],
        "top_kernels":kernel_rows_all,"phases":phases,
        "outside_engine_step_cpu_scopes_ms":ms(wall_ns-scope_union_ns),
        "outside_engine_step_cpu_scopes_whole_wall_pct":pct(wall_ns-scope_union_ns),
        "memory_operations":[{"kind":kind,"operation":label,"calls":row[0],"time_ms":ms(row[1]),"whole_wall_pct":pct(row[1]),"bytes":row[2]} for (kind,label),row in sorted(memory_groups.items())],
        "cuda_api_overlap":{
            "tables":runtime_tables,"calls":runtime_rows,"owner_thread_api_union_ms":ms(measure(api_union)),
            "owner_thread_api_overlapping_gpu_ms":ms(api_gpu_ns),"owner_thread_api_during_gpu_idle_ms":ms(measure(api_union)-api_gpu_ns),
            "owner_thread_explicit_wait_union_ms":ms(measure(wait_union)),"owner_thread_explicit_wait_overlapping_gpu_ms":ms(wait_gpu_ns),
            "owner_thread_explicit_wait_during_gpu_idle_ms":ms(measure(wait_union)-wait_gpu_ns),
            "api_duration_sums":[{"name":name,"calls":row[0],"time_ms":ms(row[1]),"whole_wall_pct":pct(row[1])} for name,row in sorted(api_groups.items(),key=lambda item:item[1][1],reverse=True)]},
        "checks":{
            "wall_conserved":sum(wall_times.values())==wall_ns,
            "kernel_calls_conserved":sum(p['kernel_calls'] for p in phases.values())==len(kernels),
            "kernel_duration_conserved":sum(group[1] for groups in phase_kernel_groups.values() for group in groups.values())==kernel_sum,
            "phase_attribution_pct":attributed_calls/len(kernels)*100,
            "phase_duration_attribution_pct":attributed_time/kernel_sum*100,
            "metadata_steps_matched":matched_metadata_steps,"metadata_step_count":len(step_metadata),
            "captured_outer_step_count":len(outer),"selected_phase_scope_count":len(scopes),
            "gpu_kernels_clipped_at_generation_boundary":clipped_kernels,"gpu_memory_operations_clipped_at_generation_boundary":memory_clipped,
            "outer_scopes_used":bool(outer),"inner_scopes_excluded_when_outer_present":bool(outer),
        },
        "warnings":warnings,
        "interpretation":[
            "wall_breakdown is an additive partition of measured_generation and sums to 100 percent. Kernel work takes precedence over overlapping copies/memsets; simultaneous distinct kernel categories use kernel:overlap.",
            "categories.time_ms, top_kernels.time_ms and phases.kernel_sum_ms are summed GPU durations. Their whole_wall_pct values are not additive when GPU activities overlap and can exceed 100 percent.",
            "categories.disjoint_wall_ms counts time with that category active and no other kernel category active. Multiple overlapping kernels of the same category count once.",
            "Phase CPU durations use complete outer engine_step_N ranges and include scheduling when capture metadata is supplied. Nested inner prefill/decode ranges are excluded.",
            "Phase GPU attribution follows CUDA correlationId to the launch API and its containing outer engine-step range, not GPU execution timestamps. Unmatched activity stays unattributed.",
            "CUDA API durations overlap GPU work and are not additional wall time. Explicit wait identifies APIs named Synchronize or Wait; other API calls can also block.",
            "gpu_idle means no activity from this captured process/device was recorded; it does not establish a cause of the gap.",
            "The generation NVTX scope excludes model loading, warmup and profiler-stop export/flush. Profiled wall time is not an unprofiled latency benchmark.",
        ],
    }
    connection.close()
    return result


def format_summary(data):
    """Return a compact wall-percentage table, without a performance judgment."""
    case = data.get("case") or {}
    title = case.get("id","capture") if isinstance(case,dict) else str(case)
    lines = [f"{title}: measured generation {data['wall_ms']/1000:.3f} s",
             "item | time (ms) | whole generation (%)"]
    for row in data["wall_breakdown"]:
        lines.append(f"{row['name']} | {row['time_ms']:.3f} | {row['whole_wall_pct']:.2f}")
    lines.append("phase CPU scopes | time (ms) | whole generation (%)")
    for phase in ("prefill","decode","other"):
        row = data["phases"][phase]
        lines.append(f"{phase} ({row['scope_count']} scopes) | {row['cpu_scope_ms']:.3f} | {row['cpu_scope_whole_wall_pct']:.2f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--capture-metadata","--metadata",dest="capture_metadata",type=Path)
    parser.add_argument("--range-name",default="measured_generation")
    parser.add_argument("--device",type=int,default=0)
    args = parser.parse_args()
    if args.output.resolve() == args.sqlite.resolve():
        parser.error("output must differ from SQLite input")
    data = analyze(args.sqlite,args.capture_metadata,args.range_name,args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(format_summary(data),flush=True)
    print("summary: " + str(args.output.resolve()),flush=True)


if __name__ == "__main__":
    main()
