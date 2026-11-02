# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the isolated extension: python -m diffkv.build_ops."""

from diffkv.ops import get_extension

if __name__ == "__main__":
    extension = get_extension(verbose=True)
    print(f"Built {extension.__file__}")
    print(f"Source and ABI fingerprint: {extension.build_fingerprint}")
