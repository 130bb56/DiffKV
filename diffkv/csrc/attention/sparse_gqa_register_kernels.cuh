/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Include inside diffkv::gqa_reuse, after the common constants are declared.
// kPartitionSize must also match sparse_paged_attention_gqa_reduce.
// QK and PV run as FP16 m16n8k16 tensor core MMAs on the exact integer codes.
// Each token's scale and zero point are applied outside the MMA:
//   logit  = scale_k * (q . code_k) + zero_k * sum(q)
//   output = sum(p * scale_v * code_v) + sum(p * zero_v)

namespace register_gqa_detail {

constexpr int kWarps = kThreads / 32;
constexpr int kTileTokens = 16;
// A partition holds at most 240 tokens. Split over its high and low regions
// it spans at most 16 tiles.
constexpr int kMaxTiles = 16;
constexpr int kTilesPerWarp = kMaxTiles / kWarps;
constexpr int kSlots = kMaxTiles * kTileTokens;
constexpr int kSlotsPerThread = kSlots / kThreads;
// The padding puts the four query rows in different banks.
constexpr int kRowStride = kSlots + 8;
constexpr int kMaxHighPages = 10;
constexpr int kMaxLowPages = 6;
// FP16 1024 in both halves. OR-ing a code below 1024 into the mantissa gives
// exactly 1024 + code.
constexpr uint32_t kMagic = 0x64006400;

static_assert(kPartitionSize <= 240 && kSlotsPerThread * kThreads == kSlots);

template <int PageTokens, int KeyBits, int ValueBits>
struct PageLayout {
  static constexpr int kTokens = PageTokens;
  static constexpr int kKeyMeta =
      DIVIDE_ROUND_UP(PageTokens * 128 * KeyBits / 16 * 2, 32) * 16;
  static constexpr int kValue =
      DIVIDE_ROUND_UP(kKeyMeta * 2 + PageTokens * 4, 128) * 64;
  static constexpr int kValueMeta =
      kValue + DIVIDE_ROUND_UP(PageTokens * 128 * ValueBits / 16 * 2, 32) * 16;
  // Distance between two value payload rows (dimension slices).
  static constexpr int kValueRow = PageTokens * 8;
};
using HighPage = PageLayout<32, 8, 4>;
using LowPage = PageLayout<60, 4, 2>;
static_assert(HighPage::kKeyMeta == 2048 && HighPage::kValue == 2112 &&
              HighPage::kValueMeta == 3136);
static_assert(LowPage::kKeyMeta == 1920 && LowPage::kValue == 2048 &&
              LowPage::kValueMeta == 3008);

// Tokens [begin, end) of one region. Tiles start at the even token
// tile_begin, so every 32-bit value load holds one stored token pair.
struct Region {
  int begin;
  int end;
  int tile_begin;
  int first_tile;
  int tiles;
  int first_page;
  int logical_offset;
};

__device__ __forceinline__ Region make_region(int begin, int end,
                                              int first_tile, int page_tokens,
                                              int logical_offset) {
  Region region;
  region.begin = begin;
  region.end = end;
  region.tile_begin = begin & ~1;
  region.first_tile = first_tile;
  region.tiles =
      end > begin ? DIVIDE_ROUND_UP(end - region.tile_begin, kTileTokens) : 0;
  region.first_page = region.tile_begin / page_tokens;
  region.logical_offset = logical_offset;
  return region;
}

__device__ __forceinline__ float sum(float value) {
#pragma unroll
  for (int delta = 16; delta; delta /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, delta);
  }
  return value;
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t hsub2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

__device__ __forceinline__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Two FP16 codes from the code bits under mask, one from each 16-bit half.
// Shift is the bit position of the code within its half.
template <int Shift>
__device__ __forceinline__ uint32_t codes(uint32_t bits, uint32_t mask) {
  const uint32_t magic = (bits & mask) | kMagic;
  if constexpr (Shift == 0) {
    return hsub2(magic, kMagic);
  } else if constexpr (Shift == 2) {
    return hfma2(magic, 0x34003400, 0xDC00DC00);  // x / 4 - 256
  } else if constexpr (Shift == 4) {
    return hfma2(magic, 0x2C002C00, 0xD400D400);  // x / 16 - 64
  } else {
    static_assert(Shift == 6);
    return hfma2(magic, 0x24002400, 0xCC00CC00);  // x / 64 - 16
  }
}

// Two FP16 codes from the 8-bit codes in bytes Selector picks.
template <uint32_t Selector>
__device__ __forceinline__ uint32_t byte_codes(uint32_t bits) {
  return hsub2(__byte_perm(bits, 0x64646464, Selector), kMagic);
}

__device__ __forceinline__ float half_sum(uint32_t bits) {
  return to_float(static_cast<uint16_t>(bits)) +
         to_float(static_cast<uint16_t>(bits >> 16));
}

// B fragments of the KV head's four queries (MMA column g, zero for g >= 4)
// in the dimension order of one key layout, and sum(q) for the C columns
// 2t and 2t + 1 this lane holds.
struct QueryFragments {
  uint32_t b[8][2];
  float sums[2];
};

__device__ __forceinline__ void finish_query_sums(float sum, int lane,
                                                  QueryFragments& query) {
  sum += __shfl_xor_sync(0xffffffff, sum, 1);
  sum += __shfl_xor_sync(0xffffffff, sum, 2);
  const int t = lane % 4;
  query.sums[0] = __shfl_sync(0xffffffff, sum, (2 * t) * 4);
  query.sums[1] = __shfl_sync(0xffffffff, sum, (2 * t + 1) * 4);
}

// Lane t holds dimensions 32a + 8t + [0, 8) for a in [0, 4). Step s takes
// the pairs 32(s / 2) + 8t + 4(s % 2) + {0, 2}, as the K8 words do.
__device__ __forceinline__ void load_high_queries(const uint16_t* queries,
                                                  QueryFragments& query) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  float total = 0.f;
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    uint4 x = make_uint4(0, 0, 0, 0);
    if (g < 4) {
      x = *reinterpret_cast<const uint4*>(queries + g * 128 + 32 * a + 8 * t);
    }
    query.b[2 * a][0] = x.x;
    query.b[2 * a][1] = x.y;
    query.b[2 * a + 1][0] = x.z;
    query.b[2 * a + 1][1] = x.w;
    total += half_sum(x.x) + half_sum(x.y) + half_sum(x.z) + half_sum(x.w);
  }
  finish_query_sums(total, lane, query);
}

// Lane t holds dimensions 32a + 16(t / 2) + [0, 16) for a = 2(t % 2) + {0, 1}.
// A K4 word pairs dimension d with d + 4, so the queries are paired the same
// way.
__device__ __forceinline__ void load_low_queries(const uint16_t* queries,
                                                 QueryFragments& query) {
  const int lane = threadIdx.x % 32;
  const int g = lane / 4;
  const int t = lane % 4;
  float total = 0.f;
#pragma unroll
  for (int chunk = 0; chunk < 2; ++chunk) {
    uint4 lo = make_uint4(0, 0, 0, 0);
    uint4 hi = lo;
    if (g < 4) {
      const uint4* source = reinterpret_cast<const uint4*>(
          queries + g * 128 + 32 * (2 * (t % 2) + chunk) + 16 * (t / 2));
      lo = source[0];
      hi = source[1];
    }
    const uint32_t y[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int i = 2 * chunk + half;
      const int k = 4 * half;
      query.b[2 * i][0] = __byte_perm(y[k], y[k + 2], 0x5410);
      query.b[2 * i][1] = __byte_perm(y[k], y[k + 2], 0x7632);
      query.b[2 * i + 1][0] = __byte_perm(y[k + 1], y[k + 3], 0x5410);
      query.b[2 * i + 1][1] = __byte_perm(y[k + 1], y[k + 3], 0x7632);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      total += half_sum(y[k]);
    }
  }
  finish_query_sums(total, lane, query);
}

// Each warp takes every fourth tile of the partition. MMA row g and g + 8 of
// a tile are its tokens g and g + 8; column 2t + e is query 2t + e.
template <bool kHigh>
__device__ __forceinline__ void compute_logits(
    const uint16_t* kv_cache, const int* pages, int page_stride,
    const Region& region, const QueryFragments& query, float scale,
    const float* alibi_slopes, int first_head, int context_length,
    float (*logits)[kRowStride], float (&maxima)[2]) {
  using Page = std::conditional_t<kHigh, HighPage, LowPage>;
  constexpr int kWords = kHigh ? 8 : 4;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int g = lane / 4;
  const int t = lane % 4;
  const int first = (warp - region.first_tile) & (kWarps - 1);

  uint32_t keys[kTilesPerWarp][2][kWords];
  uint32_t metas[kTilesPerWarp][2];
#pragma unroll
  for (int i = 0; i < kTilesPerWarp; ++i) {
    const int tile = first + i * kWarps;
#pragma unroll
    for (int row = 0; row < 2; ++row) {
      const int token = region.tile_begin + tile * kTileTokens + row * 8 + g;
#pragma unroll
      for (int word = 0; word < kWords; ++word) {
        keys[i][row][word] = 0;
      }
      metas[i][row] = 0;
      if (tile < region.tiles && token >= region.begin &&
          token < region.end) {
        const uint16_t* page =
            kv_cache +
            int64_t(pages[token / Page::kTokens - region.first_page]) *
                page_stride;
        const int slot = token % Page::kTokens;
        // K8: lane t reads dimension slice t, 32 bytes. K4: lane t reads half
        // of slice t / 2, 16 bytes.
        const uint4* key = reinterpret_cast<const uint4*>(
            page + (kHigh ? (t * 32 + slot) * 16
                          : ((t / 2) * 60 + slot) * 16 + (t % 2) * 8));
#pragma unroll
        for (int vector = 0; vector < kWords / 4; ++vector) {
          const uint4 data = key[vector];
          keys[i][row][4 * vector] = data.x;
          keys[i][row][4 * vector + 1] = data.y;
          keys[i][row][4 * vector + 2] = data.z;
          keys[i][row][4 * vector + 3] = data.w;
        }
        metas[i][row] = *reinterpret_cast<const uint32_t*>(
            page + Page::kKeyMeta + slot * 2);
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kTilesPerWarp; ++i) {
    const int tile = first + i * kWarps;
    if (tile >= region.tiles) {
      continue;
    }
    float products[4] = {0.f, 0.f, 0.f, 0.f};
    if constexpr (kHigh) {
#pragma unroll
      for (int step = 0; step < 8; ++step) {
        mma(products, byte_codes<0x4140>(keys[i][0][step]),
            byte_codes<0x4140>(keys[i][1][step]),
            byte_codes<0x4342>(keys[i][0][step]),
            byte_codes<0x4342>(keys[i][1][step]), query.b[step][0],
            query.b[step][1]);
      }
    } else {
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        uint32_t nibbles[2][4];
#pragma unroll
        for (int row = 0; row < 2; ++row) {
          const uint32_t bits = keys[i][row][word];
          nibbles[row][0] = codes<0>(bits, 0x000F000F);
          nibbles[row][1] = codes<4>(bits, 0x00F000F0);
          nibbles[row][2] = codes<0>(bits >> 8, 0x000F000F);
          nibbles[row][3] = codes<4>(bits >> 8, 0x00F000F0);
        }
        mma(products, nibbles[0][0], nibbles[1][0], nibbles[0][1],
            nibbles[1][1], query.b[2 * word][0], query.b[2 * word][1]);
        mma(products, nibbles[0][2], nibbles[1][2], nibbles[0][3],
            nibbles[1][3], query.b[2 * word + 1][0],
            query.b[2 * word + 1][1]);
      }
    }
    if (t < 2) {
#pragma unroll
      for (int row = 0; row < 2; ++row) {
        const int token = region.tile_begin + tile * kTileTokens + row * 8 + g;
        const bool valid = token >= region.begin && token < region.end;
        const float key_scale = to_float(static_cast<uint16_t>(metas[i][row]));
        const float key_zero =
            to_float(static_cast<uint16_t>(metas[i][row] >> 16));
        const int slot =
            (region.first_tile + tile) * kTileTokens + row * 8 + g;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float logit = (key_scale * products[2 * row + e] +
                         key_zero * query.sums[e]) *
                        scale;
          if (alibi_slopes != nullptr) {
            logit += alibi_slopes[first_head + 2 * t + e] *
                     (region.logical_offset + token - context_length + 1);
          }
          logits[2 * t + e][slot] = valid ? logit : -INFINITY;
          if (valid) {
            maxima[e] = fmaxf(maxima[e], logit);
          }
        }
      }
    }
  }
}

// Value words for the MMA A fragments of every tile: rows g and g + 8 are
// value dimensions of slice g, k = 2t, 2t + 1 and 2t + 8, 2t + 9 are the tile
// tokens. Warp w covers dimensions 16g + 4w + [0, 4).
__device__ __forceinline__ void load_values(
    const uint16_t* kv_cache, const int* high_pages, const int* low_pages,
    int page_stride, const Region& high, const Region& low,
    uint32_t (&values)[kMaxTiles][2]) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int g = lane / 4;
  const int t = lane % 4;
  const int tiles = high.tiles + low.tiles;
#pragma unroll
  for (int tile = 0; tile < kMaxTiles; ++tile) {
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      values[tile][pair] = 0;
    }
    if (tile >= tiles) {
      continue;
    }
    const bool is_high = tile < high.tiles;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      if (is_high) {
        const int token = high.tile_begin + tile * kTileTokens + 8 * pair +
                          2 * t;
        if (token < high.end) {
          const int slot = token % HighPage::kTokens;
          const uint16_t* page =
              kv_cache +
              int64_t(high_pages[token / HighPage::kTokens - high.first_page]) *
                  page_stride;
          values[tile][pair] = *reinterpret_cast<const uint32_t*>(
              page + HighPage::kValue + warp * HighPage::kValueRow +
              slot / 2 * 16 + 2 * g);
        }
      } else {
        const int token = low.tile_begin +
                          (tile - low.first_tile) * kTileTokens + 8 * pair +
                          2 * t;
        if (token < low.end) {
          const int slot = token % LowPage::kTokens;
          const uint16_t* page =
              kv_cache +
              int64_t(low_pages[token / LowPage::kTokens - low.first_page]) *
                  page_stride;
          values[tile][pair] = *reinterpret_cast<const uint32_t*>(
              page + LowPage::kValue + (warp / 2) * LowPage::kValueRow +
              slot / 2 * 16 + 2 * g);
        }
      }
    }
  }
}

}  // namespace register_gqa_detail

template <typename scalar_t>
__global__ __launch_bounds__(128, 4) void register_gqa_attention_kernel(
    const int* slot_ids, scalar_t* tmp_out, float* tmp_scores, float* exp_sums,
    float* max_logits, const scalar_t* queries, const uint16_t* kv_cache,
    const int* block_tables, const int* kv_len_tables, int layer_idx,
    int num_layers, int num_kv_heads, int max_blocks, int unified_page_size,
    int max_context_len, int query_stride, float scale,
    const float* alibi_slopes) {
  using namespace register_gqa_detail;
  static_assert(kThreads == 128 && kHeadSize == 128 && kQueriesPerKv == 4);
  static_assert(sizeof(scalar_t) == sizeof(uint16_t));
  const int sequence = blockIdx.y;
  const int kv_head = blockIdx.x;
  const int partition = blockIdx.z;
  const int partition_stride = gridDim.z;
  const int num_heads = num_kv_heads * 4;
  const int first_head = kv_head * 4;
  const int table_offset =
      (slot_ids[sequence] * num_layers + layer_idx) * num_kv_heads + kv_head;
  const int high_length = kv_len_tables[table_offset * 2];
  const int low_length = kv_len_tables[table_offset * 2 + 1];
  const int context_length = high_length + low_length;
  int high_begin = 0, high_end = high_length;
  int low_begin = 0, low_end = low_length;
  if (context_length > kPartitionSize) {
    const int high_partitions = DIVIDE_ROUND_UP(high_length, kPartitionSize);
    const int low_partitions = DIVIDE_ROUND_UP(low_length, kPartitionSize);
    if (partition >= high_partitions + low_partitions) {
      return;
    }
    if (partition < high_partitions) {
      high_begin = partition * kPartitionSize;
      high_end = MIN(high_begin + kPartitionSize, high_length);
      low_end = 0;
    } else {
      high_end = 0;
      low_begin = (partition - high_partitions) * kPartitionSize;
      low_end = MIN(low_begin + kPartitionSize, low_length);
    }
  } else if (partition != 0) {
    return;
  }
  const Region high =
      make_region(high_begin, high_end, 0, HighPage::kTokens, 0);
  const Region low = make_region(low_begin, low_end, high.tiles,
                                 LowPage::kTokens, high_length);
  const int tiles = high.tiles + low.tiles;
  const int* block_table = block_tables + table_offset * max_blocks;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int g = lane / 4;
  const int t = lane % 4;
  __shared__ float logits[4][kRowStride];
  __shared__ uint16_t weights[4][kRowStride];
  __shared__ int high_pages[kMaxHighPages];
  __shared__ int low_pages[kMaxLowPages];
  __shared__ float warp_maxima[4][kWarps];
  __shared__ float warp_sums[4][kWarps];
  __shared__ float warp_zeros[4][kWarps];

  if (threadIdx.x < kMaxHighPages) {
    const int page = high.first_page + threadIdx.x;
    if (high.tiles > 0 && page <= (high.end - 1) / HighPage::kTokens) {
      high_pages[threadIdx.x] = block_table[page];
    }
  } else if (threadIdx.x >= 32 && threadIdx.x < 32 + kMaxLowPages) {
    const int page = low.first_page + threadIdx.x - 32;
    if (low.tiles > 0 && page <= (low.end - 1) / LowPage::kTokens) {
      low_pages[threadIdx.x - 32] = block_table[max_blocks - 1 - page];
    }
  }
  __syncthreads();

  // Each thread later normalizes the slots threadIdx.x + k * kThreads. Fetch
  // their value metadata now, behind the key work.
  int logical_tokens[kSlotsPerThread];
  bool valid_slots[kSlotsPerThread];
  uint32_t value_metas[kSlotsPerThread];
#pragma unroll
  for (int k = 0; k < kSlotsPerThread; ++k) {
    const int slot = threadIdx.x + k * kThreads;
    const bool is_high = slot < low.first_tile * kTileTokens;
    const int token = is_high
                          ? high.tile_begin + slot
                          : low.tile_begin + slot - low.first_tile * kTileTokens;
    valid_slots[k] =
        slot < tiles * kTileTokens &&
        (is_high ? token >= high.begin && token < high.end
                 : token >= low.begin && token < low.end);
    logical_tokens[k] = is_high ? token : high_length + token;
    value_metas[k] = 0;
    if (valid_slots[k]) {
      const uint16_t* page =
          kv_cache +
          int64_t(is_high
                      ? high_pages[token / HighPage::kTokens - high.first_page]
                      : low_pages[token / LowPage::kTokens - low.first_page]) *
              unified_page_size;
      value_metas[k] = *reinterpret_cast<const uint32_t*>(
          page + (is_high ? HighPage::kValueMeta +
                                token % HighPage::kTokens * 2
                          : LowPage::kValueMeta +
                                token % LowPage::kTokens * 2));
    }
  }

  float maxima[2] = {-FLT_MAX, -FLT_MAX};
  {
    const uint16_t* query = reinterpret_cast<const uint16_t*>(queries) +
                            int64_t(sequence) * query_stride + first_head * 128;
    if (high.tiles > 0) {
      QueryFragments fragments;
      load_high_queries(query, fragments);
      compute_logits<true>(kv_cache, high_pages, unified_page_size, high,
                           fragments, scale, alibi_slopes, first_head,
                           context_length, logits, maxima);
    }
    if (low.tiles > 0) {
      QueryFragments fragments;
      load_low_queries(query, fragments);
      compute_logits<false>(kv_cache, low_pages, unified_page_size, low,
                            fragments, scale, alibi_slopes, first_head,
                            context_length, logits, maxima);
    }
  }
  // Issued before the softmax so its latency overlaps the barriers.
  uint32_t values[kMaxTiles][2];
  load_values(kv_cache, high_pages, low_pages, unified_page_size, high, low,
              values);

#pragma unroll
  for (int e = 0; e < 2; ++e) {
#pragma unroll
    for (int delta = 4; delta < 32; delta *= 2) {
      maxima[e] = fmaxf(maxima[e], __shfl_xor_sync(0xffffffff, maxima[e], delta));
    }
  }
  if (lane < 2) {
    warp_maxima[2 * lane][warp] = maxima[0];
    warp_maxima[2 * lane + 1][warp] = maxima[1];
  }
  __syncthreads();

  float maximum_logits[4];
  float sums[4] = {0.f, 0.f, 0.f, 0.f};
  float zeros[4] = {0.f, 0.f, 0.f, 0.f};
  float exponentials[kSlotsPerThread][4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    maximum_logits[q] = warp_maxima[q][0];
#pragma unroll
    for (int w = 1; w < kWarps; ++w) {
      maximum_logits[q] = fmaxf(maximum_logits[q], warp_maxima[q][w]);
    }
  }
#pragma unroll
  for (int k = 0; k < kSlotsPerThread; ++k) {
    const int slot = threadIdx.x + k * kThreads;
    const float value_scale = to_float(static_cast<uint16_t>(value_metas[k]));
    const float value_zero =
        to_float(static_cast<uint16_t>(value_metas[k] >> 16));
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float exponential =
          valid_slots[k] ? __expf(logits[q][slot] - maximum_logits[q]) : 0.f;
      exponentials[k][q] = exponential;
      sums[q] += exponential;
      zeros[q] += valid_slots[k] ? exponential * value_zero : 0.f;
      if (slot < tiles * kTileTokens) {
        from_float(weights[q][slot],
                   valid_slots[k] ? exponential * value_scale : 0.f);
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    sums[q] = sum(sums[q]);
    zeros[q] = sum(zeros[q]);
    if (lane == 0) {
      warp_sums[q][warp] = sums[q];
      warp_zeros[q][warp] = zeros[q];
    }
  }
  __syncthreads();

  float inverse_sums[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float exponential_sum = 0.f;
    zeros[q] = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      exponential_sum += warp_sums[q][w];
      zeros[q] += warp_zeros[q][w];
    }
    inverse_sums[q] = __fdividef(1.f, exponential_sum + 1e-6f);
    if (threadIdx.x == 0) {
      const int index =
          (sequence * num_heads + first_head + q) * partition_stride +
          partition;
      max_logits[index] = maximum_logits[q];
      exp_sums[index] = exponential_sum;
    }
  }
#pragma unroll
  for (int k = 0; k < kSlotsPerThread; ++k) {
    if (valid_slots[k]) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        tmp_scores[(sequence * num_heads + first_head + q) * max_context_len +
                   logical_tokens[k]] = exponentials[k][q] * inverse_sums[q];
      }
    }
  }

  float outputs[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int tile = 0; tile < kMaxTiles; ++tile) {
    if (tile >= tiles) {
      continue;
    }
    uint32_t b0 = 0, b1 = 0;
    if (g < 4) {
      b0 = *reinterpret_cast<const uint32_t*>(
          &weights[g][tile * kTileTokens + 2 * t]);
      b1 = *reinterpret_cast<const uint32_t*>(
          &weights[g][tile * kTileTokens + 8 + 2 * t]);
    }
    uint32_t a[2][4];
    if (tile < high.tiles) {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const uint32_t bits = values[tile][pair];
        a[pair][0] = codes<0>(bits, 0x000F000F);
        a[pair][1] = codes<4>(bits, 0x00F000F0);
        a[pair][2] = codes<0>(bits >> 8, 0x000F000F);
        a[pair][3] = codes<4>(bits >> 8, 0x00F000F0);
      }
    } else {
#pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const uint32_t bits = warp % 2 ? values[tile][pair] >> 8
                                       : values[tile][pair];
        a[pair][0] = codes<0>(bits, 0x00030003);
        a[pair][1] = codes<2>(bits, 0x000C000C);
        a[pair][2] = codes<4>(bits, 0x00300030);
        a[pair][3] = codes<6>(bits, 0x00C000C0);
      }
    }
    mma(outputs[0], a[0][0], a[0][1], a[1][0], a[1][1], b0, b1);
    mma(outputs[1], a[0][2], a[0][3], a[1][2], a[1][3], b0, b1);
  }

  // Lane (g, t < 2) holds dimensions 16g + 4w + 2m + {0, 1} of queries 2t and
  // 2t + 1.
  if (t < 2) {
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int dimension = 16 * g + 4 * warp + 2 * m;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int q = 2 * t + e;
        const float zero = t == 0 ? zeros[e] : zeros[2 + e];
        const float inverse = t == 0 ? inverse_sums[e] : inverse_sums[2 + e];
        uint16_t first, second;
        from_float(first, (outputs[m][e] + zero) * inverse);
        from_float(second, (outputs[m][2 + e] + zero) * inverse);
        const int index =
            ((sequence * num_heads + first_head + q) * partition_stride +
             partition) *
                128 +
            dimension;
        *reinterpret_cast<uint32_t*>(tmp_out + index) =
            uint32_t(first) | (uint32_t(second) << 16);
      }
    }
  }
}
