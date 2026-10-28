# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the isolated extension: python -m vllm.diffkv_v2.build_ops."""

from vllm.diffkv_v2.ops import get_extension

if __name__ == "__main__":
    extension = get_extension(verbose=True)
    print(f"Built {extension.__file__}")
    print(f"Source and ABI fingerprint: {extension.build_fingerprint}")
