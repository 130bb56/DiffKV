# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record nvidia-smi GPU telemetry as JSONL every 200 ms until SIGTERM or SIGINT.

Each line holds the host epoch time of the query and the queried values, so
check_throttle.py can match samples to a measured interval.
"""

import argparse
import json
import signal
import subprocess
import time

FIELDS = {
    "index": "index",
    "uuid": "uuid",
    "pstate": "pstate",
    "temperature_c": "temperature.gpu",
    "power_w": "power.draw",
    "sm_mhz": "clocks.current.sm",
    "memory_mhz": "clocks.current.memory",
    "reasons": "clocks_throttle_reasons.active",
    "memory_used_mib": "memory.used",
    "utilization_pct": "utilization.gpu",
}


def parse(key, value):
    if key in ("uuid", "pstate"):
        return value
    if key == "reasons":
        return int(value, 16)
    number = float(value)
    return int(number) if number.is_integer() else number


def sample(gpu):
    start = time.time()
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--format=csv,noheader,nounits",
            "--query-gpu=" + ",".join(FIELDS.values()),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    end = time.time()
    record = {"epoch": (start + end) / 2, "query_s": end - start}
    values = [value.strip() for value in result.stdout.split(",")]
    if result.returncode or len(values) != len(FIELDS):
        record["error"] = (result.stderr or result.stdout).strip()
        return record
    try:
        record.update({key: parse(key, value) for key, value in zip(FIELDS, values)})
    except ValueError as error:
        record["error"] = str(error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    next_time = time.monotonic()
    with open(args.output, "a") as output:
        while running:
            output.write(json.dumps(sample(args.gpu)) + "\n")
            output.flush()
            next_time += args.interval
            time.sleep(max(0.0, next_time - time.monotonic()))


if __name__ == "__main__":
    main()
