#!/usr/bin/env python3

"""
==============================================================================
Initial Codebook Generation Pipeline
==============================================================================

Purpose:
    Generates the complete initial 4-of-N codebook used as the candidate pool
    for subsequent error-correcting codebook optimization.

Core Method:
    1. Enumerates all combinations of four active positions among N total
       positions.
    2. Assigns each active position one of F possible non-zero symbols
       (1 ... F), while 0 represents an inactive position.
    3. Generates the complete set of valid 4-of-N codes.
    4. Validates the generated codebook by checking for duplicate codes,
       incorrect numbers of active positions, all-zero codes, and consistency
       with the theoretical codebook size:

           C(N, 4) × F^4

Input Parameters:
    F:
        Number of possible non-zero symbols.

    N:
        Total number of positions / rounds.

    Both F and N are specified through command-line arguments.

Example Usage:
    python generate_initial_codebook.py --F 1 --N 8

Output:
    The generated initial codebook is saved as:

        initial_codebook/codebook_F{F}N{N}.csv

    This file can be used directly as input for
    `optimize_error_correcting_codebook.py`.

==============================================================================
"""

import argparse
import csv
from itertools import combinations, product
from math import comb
from pathlib import Path


ACTIVE_POSITIONS = 4


def generate_initial_codebook(F: int, N: int):
    """
    Generate all 4-of-N codes.

    Parameters
    ----------
    F : int
        Number of possible non-zero symbols (1 ... F).
    N : int
        Total number of positions / rounds.

    Returns
    -------
    list[list[int]]
        Generated initial codebook.
    """

    if F < 1:
        raise ValueError("F must be >= 1.")

    if N < ACTIVE_POSITIONS:
        raise ValueError(
            f"N must be >= {ACTIVE_POSITIONS}. Got N={N}."
        )

    codebook = []

    # Select four active positions
    for positions in combinations(range(N), ACTIVE_POSITIONS):

        # Assign one of F symbols to each active position
        for symbols in product(
            range(1, F + 1),
            repeat=ACTIVE_POSITIONS
        ):
            code = [0] * N

            for position, symbol in zip(positions, symbols):
                code[position] = symbol

            codebook.append(code)

    return codebook


def validate_codebook(codebook, F: int, N: int):
    """
    Validate the generated initial codebook.

    Checks
    ------
    1. No duplicate codes.
    2. Each code contains exactly four non-zero positions.
    3. No all-zero code exists.
    4. Total number of codes matches C(N,4) * F^4.
    """

    # Check duplicates
    unique_codes = set(tuple(code) for code in codebook)

    if len(unique_codes) != len(codebook):
        raise ValueError("Duplicate codes were detected.")

    # Check active positions
    for code in codebook:
        non_zero_count = sum(value != 0 for value in code)

        if non_zero_count != ACTIVE_POSITIONS:
            raise ValueError(
                "A code with an incorrect number of active positions "
                "was detected."
            )

    # Check all-zero code
    if any(all(value == 0 for value in code) for code in codebook):
        raise ValueError("An all-zero code was detected.")

    # Check expected number of codes
    expected_count = (
        comb(N, ACTIVE_POSITIONS)
        * (F ** ACTIVE_POSITIONS)
    )

    if len(codebook) != expected_count:
        raise ValueError(
            f"Unexpected codebook size: "
            f"expected {expected_count}, got {len(codebook)}."
        )

    return expected_count


def write_codebook_csv(codebook, output_file: Path):
    """Write the initial codebook to CSV."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    N = len(codebook[0])

    header = [""] + [
        f"Round{i}"
        for i in range(1, N + 1)
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(header)

        for index, code in enumerate(
            codebook,
            start=1
        ):
            writer.writerow(
                [f"code{index}", *code]
            )


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a 4-of-N initial codebook "
            "for error-correcting codebook design."
        )
    )

    parser.add_argument(
        "--F",
        type=int,
        required=True,
        help="Number of non-zero symbols / fluorescence channels."
    )

    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Total number of positions / rounds."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("initial_codebook"),
        help=(
            "Output directory "
            "(default: initial_codebook)."
        )
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("========== Initial codebook generation ==========")
    print(f"F = {args.F}")
    print(f"N = {args.N}")
    print(f"Active positions = {ACTIVE_POSITIONS}")

    # Generate codebook
    codebook = generate_initial_codebook(
        F=args.F,
        N=args.N
    )

    # Validate codebook
    expected_count = validate_codebook(
        codebook=codebook,
        F=args.F,
        N=args.N
    )

    print(f"Expected number of codes = {expected_count}")
    print(f"Generated number of codes = {len(codebook)}")
    print("Codebook validation passed.")

    # Output
    output_file = (
        args.output_dir
        / f"codebook_F{args.F}N{args.N}.csv"
    )

    write_codebook_csv(
        codebook=codebook,
        output_file=output_file
    )

    print(f"Initial codebook saved to: {output_file}")


if __name__ == "__main__":
    main()