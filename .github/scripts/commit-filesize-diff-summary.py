#!/usr/bin/env python3
import argparse
import subprocess
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple


@dataclass
class GitDiffTreeRecord:
    """Represents a line of output from 'git diff-tree'"""

    src_mode: str
    src_hash: str
    dst_mode: str
    dst_hash: str
    src_path: Path
    dst_path: Optional[Path]
    status: Literal["A", "C", "D", "M", "R", "T", "U", "X"]
    score: Optional[int]


@dataclass
class GitChange:
    diff_record: GitDiffTreeRecord
    bytes_changed: int


def parse_git_diff_tree_output(output: str) -> List[GitDiffTreeRecord]:
    """Parses the output of `git diff-tree` as described in the "Raw Output" section
    of the man page
    """

    def make_record(line: str) -> GitDiffTreeRecord:
        src_mode, dst_mode, src_hash, dst_hash, rest = line[1:].split(" ", maxsplit=4)
        status_score_and_paths = rest.split("\t")
        return GitDiffTreeRecord(
            src_mode=src_mode,
            src_hash=src_hash,
            dst_mode=dst_mode,
            dst_hash=dst_hash,
            status=status_score_and_paths[0][0],
            score=int(status_score_and_paths[0][1:])
            if len(status_score_and_paths[0]) > 1
            else None,
            src_path=Path(status_score_and_paths[1]),
            dst_path=Path(status_score_and_paths[2])
            if len(status_score_and_paths) >= 3
            else None,
        )

    return [make_record(line) for line in output.splitlines(keepends=False)]


def get_blob_sizes(hashes: Iterable[str]) -> Dict[str, Optional[int]]:
    """Fetches the sizes, in bytes, of git blobs

    :param hashes: A iterable of git blob hashes
    :type hashes: Iterable[str]

    :return: A dictionary that mapping blob hashes to their size if the blob exists,
             or None otherwise
    :rtype: Dict[str, Optional[int]]
    """
    input = "\n".join(set(hashes))
    cat_file_output = subprocess.run(
        ["git", "cat-file", "--batch-check"],
        input=input,
        check=True,
        text=True,
        capture_output=True,
    ).stdout

    def make_object_size_tuple(line: str) -> Tuple[str, Optional[int]]:
        hash, *_, size = line.split()
        return (hash, int(size) if size != "missing" else None)

    return dict(
        make_object_size_tuple(line)
        for line in cat_file_output.splitlines(keepends=False)
    )


def get_file_size_differences(commit_range: str) -> Dict[Path, GitChange]:
    """Computes the size difference, in bytes, of files changed between two commits

    :param commit_range: A git commit range (e.g. HEAD~3..HEAD)
    :type commit_range: str

    :return: A dictionary mapping paths (relative to repository root) to size differences
    :rtype: dict[Path, GitChange]
    """
    changed_records = parse_git_diff_tree_output(
        subprocess.run(
            ["git", "diff-tree", "-r", commit_range],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )

    sizes = get_blob_sizes(
        chain.from_iterable((idx.src_hash, idx.dst_hash) for idx in changed_records)
    )

    assert {"A", "D", "M"}.issuperset(idx.status for idx in changed_records)

    def as_int(maybe_num: Optional[int]) -> int:
        return maybe_num or 0

    return {
        x.src_path: GitChange(
            diff_record=x,
            bytes_changed=as_int(sizes[x.dst_hash]) - as_int(sizes[x.src_hash]),
        )
        for x in changed_records
    }


def main(
    commit_range: str,
    quiet: bool = False,
    limit: Optional[int] = None,
    show_n_largest_files: int = 30,
) -> Literal[0, 1]:
    size_differences = get_file_size_differences(commit_range)
    cumulative_size_difference = sum(x.bytes_changed for x in size_differences.values())
    exceeds_limit = limit is not None and cumulative_size_difference > limit

    def bytes_diff(num):
        return ("+" if num >= 0 else "") + human_friendly_bytes(num)

    if not quiet:
        print(f"Total file size difference for commit range '{commit_range}': ")
        print(f"\t{bytes_diff(cumulative_size_difference)}", end="")
        print(f" (Exceeds set limit of {bytes_diff(limit)})" if exceeds_limit else "")

        largest_n_sizes = sorted(
            size_differences.items(), key=lambda x: x[1].bytes_changed, reverse=True
        )[:show_n_largest_files]

        if largest_n_sizes:
            print("")
            print(f"Largest {len(largest_n_sizes)} filesize differences:")

        for path, val in largest_n_sizes:
            print(f"\t{bytes_diff(val.bytes_changed)}\t{path}")

    return 1 if exceeds_limit else 0


def num_bytes(arg: str) -> int:
    """Converts a string to a number of bytes"""
    error = argparse.ArgumentTypeError(
        f"'{arg}' cannot be parsed into a number of bytes"
    )
    try:
        return int(arg)
    except ValueError:
        pass

    if len(arg) < 3:
        raise error

    num, suffix = (arg[:-2], arg[-2:])
    shift_values = {
        "KB": 1,
        "MB": 2,
        "GB": 3,
        "TB": 4,
        "PB": 5,
        "EB": 6,
        "ZB": 7,
        "YB": 8,
    }

    shift = shift_values.get(suffix, None)

    if shift is None:
        raise error
    try:
        return int(num) << (shift * 10)
    except ValueError:
        raise error


def human_friendly_bytes(num: int) -> str:
    """Prints a number of bytes as a human friendly string"""
    for prefix in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:.1f}{prefix}B"
        num /= 1024.0
    return f"{num:.1f}YB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program that summarizes the file size "
        + "differences between two git commits."
    )
    parser.add_argument(
        "commit_range",
        default="HEAD^..HEAD",
        type=str,
        help="Commit range to commit the diff for (e.g. HEAD~3..HEAD)",
    )
    parser.add_argument("--quiet", action="store_true", help="Silence all output")
    parser.add_argument(
        "--limit",
        type=num_bytes,
        help="Exit non-zero if total changes exceeds this value. "
        + "Can be a raw number of bytes (e.g. 65536) or a suffixed value (e.g 2MB)",
    )
    parser.add_argument(
        "--show-n-largest-files",
        type=int,
        help="Show this many of the largest files in diff",
        default=30,
    )
    sys.exit(main(**vars(parser.parse_args())))
