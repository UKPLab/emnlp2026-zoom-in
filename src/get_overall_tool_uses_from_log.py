import argparse
import ast
import re
from collections import Counter


PATTERN = re.compile(r"overall_tools_used:\s*(\[[^\]]*\])")


def parse_overall_tools_used(line: str) -> list[int] | None:
    """
    Extract and parse the list from a line like:
      ... overall_tools_used: [1, 0, 2]
    Returns None if the line doesn't contain the field.
    """
    m = PATTERN.search(line)
    if not m:
        return None

    raw = m.group(1)
    try:
        value = ast.literal_eval(raw)  # safe for Python literals like lists of ints
    except (SyntaxError, ValueError):
        return None

    if not isinstance(value, list):
        return None

    out: list[int] = []
    for x in value:
        # Be forgiving: accept ints or strings like "1"
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return None
    return out


def aggregate_log(path: str, *, encoding: str = "utf-8") -> dict:
    total_sum = 0
    total_count = 0
    line_matches = 0
    hist = Counter()

    with open(path, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            vals = parse_overall_tools_used(line)
            if vals is None:
                continue
            line_matches += 1
            total_sum += sum(vals)
            total_count += len(vals)
            hist.update(vals)

    mean = (total_sum / total_count) if total_count else 0.0
    return {
        "file": path,
        "matched_lines": line_matches,
        "num_entries": total_count,
        "total_tools_used": total_sum,
        "mean_tools_used_per_entry": mean,
        "histogram": dict(sorted(hist.items(), key=lambda kv: kv[0])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate overall_tools_used from a log file.")
    ap.add_argument("log_path", help="Path to the log file")
    ap.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    args = ap.parse_args()

    res = aggregate_log(args.log_path, encoding=args.encoding)

    print(f"File: {res['file']}")
    print(f"Matched lines: {res['matched_lines']}")
    print(f"Total entries (sum of list lengths): {res['num_entries']}")
    print(f"Total tools used (sum of all numbers): {res['total_tools_used']}")
    print(f"Mean tools used per entry: {res['mean_tools_used_per_entry']:.6f}")
    print("Histogram (tools_used_value -> count):")
    for k, v in res["histogram"].items():
        print(f"  {k} -> {v}")


if __name__ == "__main__":
    main()