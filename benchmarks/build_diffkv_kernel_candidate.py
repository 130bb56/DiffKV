"""Build isolated, restricted attention specializations for fast kernel tuning.

Screening binaries accept only D128, GQA4, K8V4/K4V2 with one chunk. They must
not replace the normal extension: rebuild the full dispatch after selecting a
candidate. Existing non-attention objects are reused without modification.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--legacy-partition', type=int, default=5760)
parser.add_argument('--legacy-threads', type=int, default=256)
parser.add_argument('--gqa-partition', type=int, default=960)
parser.add_argument('--gqa-tile', type=int, default=32)
parser.add_argument('--gqa-source', type=Path)
parser.add_argument('--stream-qk', action='store_true')
parser.add_argument('--transpose-query', action='store_true')
parser.add_argument('--register-gqa', action='store_true')
parser.add_argument('--register-header', type=Path)
parser.add_argument('--gqa-threads', type=int,
                    help='Defaults to the selected GQA source thread count')
parser.add_argument('--reuse-main', type=Path)
parser.add_argument('--gqa-ctas-per-kv', type=int, default=1)
parser.add_argument('--no-carveout', action='store_true')
parser.add_argument('--fused-reduce', action='store_true')
parser.add_argument('--fused-reduce-header', type=Path)
parser.add_argument('--reduce-splits', type=int,
                    help='Score-update CTAs per KV head; defaults to the selected source or one')
parser.add_argument('--render-only', action='store_true',
                    help='Copy and render sources and provenance without compiling or linking')
args = parser.parse_args()
if args.reduce_splits is not None and args.reduce_splits < 1:
    parser.error('--reduce-splits must be positive')
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=False)
temp = ROOT / 'build/temp.linux-x86_64-cpython-311'
main_path = ROOT / 'csrc/attention/sparse_attention_kernels.cu'
gqa_path = ROOT / 'csrc/attention/sparse_gqa_attention_kernels.cu'
main = main_path.read_text()
start = main.index('  // Dispatch the kernel based on the head size')
end = main.index('\n#define CALL_LAUNCHER(', start)
main = main[:start] + '''  TORCH_CHECK(head_size == 128 && num_queries_per_kv == 4,
      "Restricted tuning binary requires D128 and GQA4");
  LAUNCH_SPARSE_PAGED_ATTENTION(128, 4);
}
''' + main[end:]
for dtype in ('uint16_t', '__nv_bfloat16'):
    main = main.replace(f'CALL_LAUNCHER_QUANT_CONFIG({dtype});',
        'TORCH_CHECK((quant_config == std::vector<int>{8,4,4,2,1,1,1,1}), '
        '"Restricted tuning binary requires K8V4/K4V2 with one chunk");\n'
        f'    CALL_LAUNCHER({dtype}, 8,4,4,2,1,1,1,1,32,60,8);')
main = main.replace('#define NUM_THREADS 256', f'#define NUM_THREADS {args.legacy_threads}')
if args.stream_qk:
    main = main.replace('float k_floats[NUM_ELEMS_PER_THREAD];', 'float qk_partial = 0.f;')
    for precision in ('HIGH', 'LOW'):
        pattern = (r'unpack_and_dequant\(k_vec.data\[j\],\s*BITS_K_' + precision
                   + r',\s*k_quant_scale,\s*k_quant_zero_point,\s*'
                   + r'&k_floats\[\(i \* K_VEC_SIZE \+ j\) \* K_PACK_SIZE_' + precision + r'\]\);')
        replacement = f'''float decoded[K_PACK_SIZE_{precision}];
            unpack_and_dequant(k_vec.data[j], BITS_K_{precision},
                               k_quant_scale, k_quant_zero_point, decoded);
            #pragma unroll
            for (int element = 0; element < K_PACK_SIZE_{precision}; ++element) {{
              const int dimension = (i * K_VEC_SIZE + j) * K_PACK_SIZE_{precision} + element;
              if (dimension == 0) qk_partial = q_floats[thread_group_offset_k][dimension] * decoded[element];
              else qk_partial = fmaf(q_floats[thread_group_offset_k][dimension], decoded[element], qk_partial);
            }}'''
        main, count = re.subn(pattern, lambda _: replacement, main)
        assert count == 1, (precision, count)
    main = main.replace('float qk = scale * Qk_dot<THREAD_GROUP_SIZE_K>::dot(q_floats[thread_group_offset_k], k_floats);',
        '''#pragma unroll
      for (int delta = WARP_SIZE / THREAD_GROUP_SIZE_K; delta <= WARP_SIZE / 2; delta *= 2)
        qk_partial += VLLM_SHFL_DOWN_SYNC(qk_partial, delta);
      float qk = scale * qk_partial;''')
if args.transpose_query:
    assert args.stream_qk, '--transpose-query requires --stream-qk'
    main = main.replace('q_floats[THREAD_GROUP_SIZE_K][NUM_ELEMS_PER_THREAD]',
                        'q_floats[NUM_ELEMS_PER_THREAD][THREAD_GROUP_SIZE_K]')
    main = main.replace('q_floats[i / NUM_ELEMS_PER_THREAD][i % NUM_ELEMS_PER_THREAD]',
                        'q_floats[i % NUM_ELEMS_PER_THREAD][i / NUM_ELEMS_PER_THREAD]')
    main = main.replace('q_floats[thread_group_offset_k][dimension]',
                        'q_floats[dimension][thread_group_offset_k]')
main = re.sub(r'constexpr int PARTITION_SIZE = DIVIDE_ROUND_UP\(_PARTITION_SIZE,.*?;',
              f'constexpr int PARTITION_SIZE = {args.legacy_partition};', main, count=1, flags=re.S)
start = main.index('void sparse_paged_attention_gqa_reduce(')
main = main[:start] + re.sub(r'constexpr int PARTITION_SIZE = \d+;',
    f'constexpr int PARTITION_SIZE = {args.gqa_partition};', main[start:], count=1)
gqa_input = args.gqa_source.resolve() if args.gqa_source else gqa_path
gqa_source = gqa_input.read_text()
if args.gqa_threads is None:
    args.gqa_threads = int(re.search(r'constexpr int kThreads = (\d+);', gqa_source).group(1))
gqa = re.sub(r'constexpr int kPartitionSize = \d+;',
             f'constexpr int kPartitionSize = {args.gqa_partition};', gqa_source, count=1)
gqa = re.sub(r'constexpr int kTileTokens = \d+;',
             f'constexpr int kTileTokens = {args.gqa_tile};', gqa, count=1)
gqa = re.sub(r'constexpr int kThreads = \d+;',
             f'constexpr int kThreads = {args.gqa_threads};', gqa, count=1)
gqa = gqa.replace('const dim3 grid(num_kv_heads, query.size(0),',
                  f'const dim3 grid(num_kv_heads * {args.gqa_ctas_per_kv}, query.size(0),')
if args.no_carveout:
    gqa, count = re.subn(r'#ifndef USE_ROCM\n  // The tile occupies.*?#endif\n', '', gqa, count=1, flags=re.S)
    assert count == 1 or 'cudaFuncAttributePreferredSharedMemoryCarveout' not in gqa
register_include = '#include "sparse_gqa_register_kernels.cuh"'
register_gqa = args.register_gqa or register_include in gqa
if register_gqa:
    gqa = gqa.replace('static_assert(kPartitionSize % 32 == 0 && kPartitionSize % 60 == 0);',
                      'static_assert(kPartitionSize > 0);')
    adjacent_header = gqa_input.parent / 'sparse_gqa_register_kernels.cuh'
    header_path = (args.register_header or
                   (adjacent_header if register_include in gqa and adjacent_header.is_file()
                    else ROOT / 'csrc/attention/sparse_gqa_register_kernels.cuh')).resolve()
    header = header_path.read_text().replace('kWarps = 256 / 32', 'kWarps = kThreads / 32')
    header = header.replace('static_assert(kPartitionSize % 32 == 0 && kPartitionSize % 60 == 0);',
                            'static_assert(kPartitionSize > 0);')
    header = header.replace('kThreads == 256 &&', '(kThreads == 256 || kThreads == 128) &&')
    (out / 'sparse_gqa_register_kernels.cuh').write_text(header)
    marker = 'template<typename scalar_t>\nvoid launch('
    assert marker in gqa
    if register_include not in gqa:
        gqa = gqa.replace(marker, register_include + '\n\n' + marker, 1)
    gqa = gqa.replace('sparse_gqa_attention_kernel<scalar_t>', 'register_gqa_attention_kernel<scalar_t>')
reduce_include = '#include "sparse_gqa_reduce_kernels.cuh"'
fused_reduce = args.fused_reduce or reduce_include in gqa
reduce_helper = re.search(r'\bvoid launch_reduce\s*\(', gqa)
if args.reduce_splits is None:
    existing_splits = re.search(r'const dim3 grid\(num_kv_heads, out.size\(0\), (\d+)\);', gqa)
    args.reduce_splits = int(existing_splits.group(1)) if existing_splits else 1
if args.reduce_splits != 1 and not fused_reduce:
    parser.error('--reduce-splits greater than one requires a reducer include or --fused-reduce')
if fused_reduce:
    assert register_gqa
    reduce_header = args.fused_reduce_header
    if reduce_header is None:
        adjacent_reduce = gqa_input.parent / 'sparse_gqa_reduce_kernels.cuh'
        reduce_header = (adjacent_reduce if reduce_include in gqa and adjacent_reduce.is_file()
                         else ROOT / 'csrc/attention/sparse_gqa_reduce_kernels.cuh')
        if not reduce_header.is_file():
            reduce_header = ROOT / 'logs/performance/kernel-2x-20261024/rejected_fused_reduce.cuh'
    reduce_header = reduce_header.resolve()
    if not reduce_header.is_file():
        raise FileNotFoundError('Fused-reduce experiment requires --fused-reduce-header PATH')
    reduce_header_name = 'sparse_gqa_reduce_kernels.cuh'
    (out / reduce_header_name).write_text(reduce_header.read_text())
    if reduce_helper:
        begin = reduce_helper.start()
        end = gqa.index('template<typename scalar_t>\nvoid launch(', begin)
        helper = re.sub(r'constexpr int PARTITION_SIZE = \d+;',
                        f'constexpr int PARTITION_SIZE = {args.gqa_partition};', gqa[begin:end], count=1)
        helper, count = re.subn(r'const dim3 grid\(num_kv_heads, out.size\(0\)(?:, \d+)?\);',
                               f'const dim3 grid(num_kv_heads, out.size(0), {args.reduce_splits});', helper, count=1)
        assert count == 1, 'Existing launch_reduce grid was not recognized'
        gqa = gqa[:begin] + helper + gqa[end:]
    else:
        begin = main.index('void sparse_paged_attention_gqa_reduce(')
        end = main.index('\n#undef WARP_SIZE', begin)
        helper = main[begin:end].replace('void sparse_paged_attention_gqa_reduce(', 'void launch_reduce(')
        helper = helper.replace('const dim3 block(NUM_THREADS);', 'const dim3 block(256);')
        helper = helper.replace('const dim3 grid(num_kv_heads, out.size(0));',
                                f'const dim3 grid(num_kv_heads, out.size(0), {args.reduce_splits});')
        helper = helper.replace('2 * max_num_partitions * sizeof(float)', '4 * max_num_partitions * sizeof(float)')
        helper = re.sub(r'vllm::reduce_kernel<T,.*?PARTITION_SIZE>', 'register_gqa_reduce_kernel<T>', helper)
        marker = register_include
        gqa = gqa.replace(marker, marker + '\n\n' + helper)
    if reduce_include not in gqa:
        gqa = gqa.replace(register_include, register_include + '\n' + reduce_include, 1)
    gqa = gqa.replace('  sparse_paged_attention_gqa_reduce(', '  vllm::gqa_reuse::launch_reduce(')
manifest = {'experimental_only': True, 'arguments': {**vars(args), 'output_dir': str(out),
            'gqa_source': str(gqa_input), 'register_header': str(args.register_header),
            'reuse_main': str(args.reuse_main),
            'fused_reduce_header': str(args.fused_reduce_header)},
            'register_gqa': register_gqa,
            'fused_reduce': fused_reduce,
            'commands': [], 'source_sha256': {}, 'source_input_sha256': {
                str(gqa_input): hashlib.sha256(gqa_source.encode()).hexdigest()}}
if register_gqa:
    manifest['source_sha256'][str(out / 'sparse_gqa_register_kernels.cuh')] = hashlib.sha256(header.encode()).hexdigest()
    manifest['source_input_sha256'][str(header_path)] = hashlib.sha256(header_path.read_bytes()).hexdigest()
if fused_reduce:
    manifest['source_sha256'][str(out / reduce_header_name)] = hashlib.sha256(reduce_header.read_bytes()).hexdigest()
    manifest['source_input_sha256'][str(reduce_header)] = hashlib.sha256(reduce_header.read_bytes()).hexdigest()
replacements = {}
jobs = []
for source_path, source in ((main_path, main), (gqa_path, gqa)):
    copied = out / source_path.name
    copied.write_text(source)
    manifest['source_sha256'][str(copied)] = hashlib.sha256(copied.read_bytes()).hexdigest()
    if source_path == main_path:
        manifest['source_input_sha256'][str(source_path)] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if args.render_only:
        continue
    original_object = temp / 'csrc/attention' / (source_path.stem + '.o')
    obj = out / (source_path.stem + '.o')
    if source_path == main_path and args.reuse_main:
        previous_source = args.reuse_main / source_path.name
        assert previous_source.read_text() == source, 'Reused main source must be identical'
        reused_object = args.reuse_main / (source_path.stem + '.o')
        if not reused_object.is_file():
            previous_manifest = json.loads((args.reuse_main / 'build-manifest.json').read_text())
            for command in reversed(previous_manifest['commands']):
                matches = [Path(part) for part in command
                           if part.endswith('/' + source_path.stem + '.o') and Path(part).is_file()]
                if matches:
                    reused_object = matches[0]
                    break
        if not reused_object.is_file():
            raise FileNotFoundError(f'Reused attention object is unavailable: {reused_object}')
        replacements[str(original_object)] = str(reused_object.resolve())
        manifest['source_sha256'][str(copied)] = hashlib.sha256(copied.read_bytes()).hexdigest()
        continue
    command = shlex.split(subprocess.check_output(
        ['ninja', '-C', str(temp), '-t', 'commands', str(original_object)], text=True).strip())
    command = [str(copied) if part == str(source_path) else
               str(obj) if part == str(original_object) else
               str(obj) + '.d' if part == str(original_object) + '.d' else part for part in command]
    command += ['-I' + str(ROOT / 'csrc/attention'), '-lineinfo']
    manifest['commands'].append(command)
    manifest['source_sha256'][str(copied)] = hashlib.sha256(copied.read_bytes()).hexdigest()
    replacements[str(original_object)] = str(obj)
    jobs.append((command, out / (source_path.stem + '.build.log')))

if args.render_only:
    (out / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out, flush=True)
    raise SystemExit(0)

def compile_one(job):
    command, log = job
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(compile_one, jobs))
link_line = (ROOT / 'logs/performance/kernel-wide-load-20261024/k64-v32/build.log').read_text().splitlines()[-1]
link = [replacements.get(part, part) for part in shlex.split(link_line)]
binary = out / '_C.cpython-311-x86_64-linux-gnu.so'
link[link.index('-o') + 1] = str(binary)
manifest['commands'].append(link)
subprocess.run(link, cwd=ROOT, check=True)
manifest['extension_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
(out / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(binary, flush=True)
