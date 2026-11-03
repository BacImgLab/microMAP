#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================================
PacBio 16S ASV Processing and Database Construction Pipeline
==============================================================================
Purpose:
    Processes PacBio full-length 16S ASV sequences to generate a
    taxonomy-annotated and abundance-linked 16S sequence database for
    downstream probe design.

Core Method:
    Performs top-hit BLASTN annotation against a local NCBI 16S database,
    orients ASV sequences according to BLAST alignment direction, retrieves
    taxonomic lineages using TaxonKit, and filters low-confidence assignments
    based on sequence identity and query coverage. ASV taxonomy, abundance,
    and sequence information are then integrated to generate genus/species
    abundance summaries and a global FASTA database.

Key Design Features:
    1. Preflight Validation: Checks input files, FASTA/ASV consistency,
       required external tools (BLASTN, TaxonKit, csvtk), and local taxonomy
       databases before starting the analysis.

    2. Conservative Taxonomy Assignment: ASVs below the predefined BLAST
       identity or query-coverage thresholds are labeled as "unassigned"
       rather than removed from the dataset.

    3. Background Sequence Retention: Unassigned ASVs are retained in the
       global 16S FASTA database to support downstream off-target screening
       during probe design.

Example Usage:
    # Check inputs, databases, and dependencies without running the pipeline
    python pacbio_16S_pipeline.py \
        --config config/params.json \
        --check-only

    # Run the complete pipeline
    python pacbio_16S_pipeline.py \
        --config config/params.json

Outputs:
    - Species_summary_abundance.csv
    - Genus_summary_abundance.csv
    - global_all_taxa_consensus.fasta
==============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


# ============================================================
# Python dependency check before third-party imports
# ============================================================

PYTHON_DEPENDENCIES = {
    "pandas": "pandas",
    "Bio": "biopython",
}

_missing_python_packages = [
    package_name
    for module_name, package_name in PYTHON_DEPENDENCIES.items()
    if importlib.util.find_spec(module_name) is None
]

if _missing_python_packages:
    raise SystemExit(
        "[ERROR] Missing required Python package(s): "
        + ", ".join(_missing_python_packages)
        + "\nInstall them before running the pipeline, for example:\n"
        + "  pip install pandas biopython"
    )

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


# ============================================================
# Analysis constants
# ============================================================

# Taxonomy assignment thresholds.
# ASVs below either threshold are retained but labeled as unassigned.
PIDENT_THRESHOLD = 85.0
QCOVHSP_THRESHOLD = 90.0


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    base_dir: Path
    input_fasta_path: Path
    feature_table_path: Path
    blast_db: Path

    blastn_bin: str = "blastn"
    taxonkit_bin: str = "taxonkit"
    csvtk_bin: str = "csvtk"

    num_threads: int = 8

    # Keep unassigned sequences in FASTA database for off-target screening.
    export_unassigned_fasta: bool = True

    @property
    def blast_dir(self) -> Path:
        return self.base_dir / "blast"

    @property
    def oriented_dir(self) -> Path:
        return self.base_dir / "oriented"

    @property
    def taxon_dir(self) -> Path:
        return self.base_dir / "blast_tax"

    @property
    def abundance_dir(self) -> Path:
        return self.base_dir / "table_abundance"

    @property
    def species_fasta_dir(self) -> Path:
        # Historical folder name kept for compatibility with downstream scripts.
        return self.base_dir / "fin_clutser_sequence"


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in config file: {config_path}\n"
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    required_keys = [
        "base_dir",
        "input_fasta_path",
        "feature_table_path",
        "blast_db",
    ]
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    num_threads = int(params.get("num_threads", 8))
    if num_threads < 1:
        raise ValueError("'num_threads' must be >= 1")


    return PipelineConfig(
        base_dir=Path(params["base_dir"]),
        input_fasta_path=Path(params["input_fasta_path"]),
        feature_table_path=Path(params["feature_table_path"]),
        blast_db=Path(params["blast_db"]),
        blastn_bin=str(params.get("blastn_bin", "blastn")),
        taxonkit_bin=str(params.get("taxonkit_bin", "taxonkit")),
        csvtk_bin=str(params.get("csvtk_bin", "csvtk")),
        num_threads=num_threads,
        export_unassigned_fasta=bool(params.get("export_unassigned_fasta", True)),
    )


BLAST_COLUMNS = [
    "qseqid", "sseqid", "pident", "qcovhsp", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "staxids",
]

TAXON_COLUMNS = [
    "taxid", "kingdom", "phylum", "class", "order", "Family", "Genus", "Species",
]

FINAL_TAXON_COLUMNS = BLAST_COLUMNS + TAXON_COLUMNS

FASTA_SUFFIXES = {".fasta", ".fa", ".fna"}


# ============================================================
# General utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_fasta_files(input_path: Path) -> List[Path]:
    """Return one FASTA file or all FASTA files in a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in FASTA_SUFFIXES:
            raise ValueError(
                f"Unsupported FASTA suffix: {input_path.suffix}. "
                f"Allowed: {sorted(FASTA_SUFFIXES)}"
            )
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in FASTA_SUFFIXES
        )

    return []


def sample_name_from_fasta(fasta_path: Path) -> str:
    return fasta_path.stem


def sample_name_from_oriented_fasta(path: Path) -> str:
    return path.name.replace("_oriented.fasta", "")


def sample_name_from_taxonomy(path: Path) -> str:
    return path.name.replace("_blast_combined.txt", "")


def clean_parentheses(text: object) -> object:
    if pd.isna(text):
        return text
    return re.sub(r"\s*\(.*?\)", "", str(text))


def clean_taxon_name(text: object) -> str:
    """Clean taxonomy names for FASTA IDs and filenames."""
    if pd.isna(text) or str(text).strip() == "":
        return "unassigned"

    s = clean_parentheses(str(text)).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return "unassigned"

    s = s.replace("[", "").replace("]", "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unassigned"


def asv_numeric_id(asv: object) -> int:
    m = re.search(r"(\d+)$", str(asv))
    if m:
        return int(m.group(1))
    return 10**12


def wrap_sequence(seq: str, width: int = 80) -> str:
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def run_command(
    cmd: Union[Sequence[str], str],
    stdout_path: Optional[Path] = None,
    shell: bool = False,
) -> None:
    if stdout_path is None:
        result = subprocess.run(
            cmd,
            text=True,
            stderr=subprocess.PIPE,
            shell=shell,
        )
    else:
        with open(stdout_path, "w", encoding="utf-8") as fout:
            result = subprocess.run(
                cmd,
                stdout=fout,
                text=True,
                stderr=subprocess.PIPE,
                shell=shell,
            )

    if result.returncode != 0:
        cmd_text = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
        raise RuntimeError(
            f"Command failed:\n{cmd_text}\n\nSTDERR:\n{result.stderr}"
        )


def resolve_executable(command: str) -> Optional[Path]:
    """Resolve either a command in PATH or an explicit executable path."""
    command_path = Path(command).expanduser()

    # Explicit path: contains a directory component or is absolute.
    if command_path.is_absolute() or command_path.parent != Path("."):
        if command_path.exists() and command_path.is_file() and os.access(command_path, os.X_OK):
            return command_path.resolve()
        return None

    resolved = shutil.which(command)
    return Path(resolved).resolve() if resolved else None


def get_tool_version(executable: Path, tool_name: str) -> str:
    version_commands = {
        "blastn": [[str(executable), "-version"]],
        "taxonkit": [[str(executable), "version"], [str(executable), "--version"]],
        "csvtk": [[str(executable), "version"], [str(executable), "--version"]],
    }

    for cmd in version_commands.get(tool_name, []):
        try:
            result = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        text = (result.stdout or result.stderr).strip()
        if result.returncode == 0 and text:
            return text.splitlines()[0]

    return "version unavailable"


def check_external_tool(command: str, tool_name: str) -> Path:
    resolved = resolve_executable(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Required executable not found: {command}\n"
            f"Please install {tool_name} or provide its executable path in params.json."
        )

    version = get_tool_version(resolved, tool_name)
    print(f"[OK] {tool_name}: {resolved} ({version})")
    return resolved


def check_taxonkit_database(taxonkit_executable: Path) -> None:
    """Check whether TaxonKit can access its local taxonomy database."""
    try:
        result = subprocess.run(
            [str(taxonkit_executable), "lineage"],
            input="562\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "TaxonKit executable was found, but taxonomy database check timed out."
        ) from exc

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "TaxonKit is installed, but `taxonkit lineage` could not query its "
            "taxonomy database.\n"
            "Please make sure the NCBI taxonomy files required by TaxonKit are "
            "installed in the location expected by your TaxonKit setup.\n\n"
            f"TaxonKit STDERR:\n{result.stderr.strip()}"
        )

    print("[OK] TaxonKit taxonomy database: functional")


def blast_db_core_files(blast_db: Path) -> Tuple[bool, List[Path]]:
    """Check whether a nucleotide BLAST database exists for a database prefix."""
    parent = blast_db.parent
    prefix = blast_db.name

    direct = [Path(f"{blast_db}.nhr"), Path(f"{blast_db}.nin"), Path(f"{blast_db}.nsq")]
    if all(p.exists() for p in direct):
        return True, direct

    alias = Path(f"{blast_db}.nal")
    if alias.exists():
        return True, [alias]

    nhr_parts = sorted(parent.glob(f"{prefix}.*.nhr")) if parent.exists() else []
    nin_parts = sorted(parent.glob(f"{prefix}.*.nin")) if parent.exists() else []
    nsq_parts = sorted(parent.glob(f"{prefix}.*.nsq")) if parent.exists() else []

    if nhr_parts and nin_parts and nsq_parts:
        return True, nhr_parts + nin_parts + nsq_parts

    found = sorted(parent.glob(f"{prefix}.*")) if parent.exists() else []
    return False, found


def validate_blast_database(blast_db: Path, blastn_executable: Path) -> None:
    if not blast_db.parent.exists():
        raise FileNotFoundError(
            f"BLAST database directory not found: {blast_db.parent}"
        )

    valid, files = blast_db_core_files(blast_db)
    if not valid:
        found_text = "\n".join(f"  - {p.name}" for p in files[:20]) or "  (none)"
        raise FileNotFoundError(
            f"BLAST nucleotide database not found or incomplete for prefix:\n"
            f"  {blast_db}\n"
            f"Expected prefix.nhr/prefix.nin/prefix.nsq, a .nal alias, or split volumes.\n"
            f"Files found with this prefix:\n{found_text}"
        )

    print(f"[OK] BLAST DB prefix: {blast_db}")
    print(f"[OK] BLAST DB core files detected: {len(files)}")

    # Optional stronger validation using blastdbcmd if BLAST+ provides it.
    blastdbcmd_candidates = [blastn_executable.with_name("blastdbcmd")]
    path_candidate = shutil.which("blastdbcmd")
    if path_candidate:
        blastdbcmd_candidates.append(Path(path_candidate))

    blastdbcmd = next(
        (
            p.resolve()
            for p in blastdbcmd_candidates
            if p.exists() and p.is_file() and os.access(p, os.X_OK)
        ),
        None,
    )

    if blastdbcmd is not None:
        result = subprocess.run(
            [str(blastdbcmd), "-db", str(blast_db), "-info"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "BLAST database files were found, but blastdbcmd could not open "
                f"the database prefix: {blast_db}\n\n"
                f"STDERR:\n{result.stderr.strip()}"
            )
        print(f"[OK] BLAST DB opened successfully with blastdbcmd: {blastdbcmd}")
    else:
        print("[WARN] blastdbcmd not found; skipped functional BLAST DB validation.")

    # These are useful BLAST taxonomy support files, but they are not the
    # TaxonKit taxonomy database used later in this pipeline.
    taxdb_btd = blast_db.parent / "taxdb.btd"
    taxdb_bti = blast_db.parent / "taxdb.bti"
    if taxdb_btd.exists() and taxdb_bti.exists():
        print("[OK] BLAST taxonomy support files: taxdb.btd / taxdb.bti")
    else:
        print(
            "[INFO] taxdb.btd/taxdb.bti were not both detected. "
            "They are separate from the TaxonKit taxonomy database."
        )

    taxonomy_sqlite = blast_db.parent / "taxonomy4blast.sqlite3"
    if taxonomy_sqlite.exists():
        print(f"[OK] Additional BLAST taxonomy file: {taxonomy_sqlite.name}")


def read_feature_table(feature_table_path: Path) -> Tuple[pd.DataFrame, Tuple[str, ...]]:
    """Read abundance table and infer abundance columns."""
    try:
        df = pd.read_csv(feature_table_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Feature table is empty: {feature_table_path}") from exc

    if df.empty:
        raise ValueError(f"Feature table contains no data rows: {feature_table_path}")

    if "ASV" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "ASV"})
        else:
            raise ValueError(
                f"Feature table must contain a column named 'ASV'.\n"
                f"Available columns: {list(df.columns)}"
            )

    df["ASV"] = df["ASV"].astype(str).str.strip()

    if (df["ASV"] == "").any():
        raise ValueError("Feature table contains blank ASV IDs.")

    duplicated = int(df["ASV"].duplicated().sum())
    if duplicated > 0:
        duplicate_ids = df.loc[df["ASV"].duplicated(keep=False), "ASV"].unique()[:10]
        raise ValueError(
            f"Feature table contains duplicated ASV rows: {duplicated}\n"
            f"Examples: {', '.join(map(str, duplicate_ids))}"
        )

    abundance_columns = tuple(col for col in df.columns if col != "ASV")
    if not abundance_columns:
        raise ValueError(
            "Feature table must contain at least one abundance column in addition to 'ASV'."
        )

    for col in abundance_columns:
        original = df[col]
        numeric = pd.to_numeric(original, errors="coerce")

        invalid_mask = original.notna() & numeric.isna()
        if invalid_mask.any():
            examples = original[invalid_mask].astype(str).head(5).tolist()
            raise ValueError(
                f"Abundance column '{col}' contains non-numeric value(s): {examples}"
            )

        numeric = numeric.fillna(0)
        if (numeric < 0).any():
            raise ValueError(
                f"Abundance column '{col}' contains negative values, which are not allowed."
            )

        df[col] = numeric

    print(f"[OK] Abundance columns inferred: {', '.join(abundance_columns)}")
    return df, abundance_columns


def validate_fasta_inputs(fasta_files: Sequence[Path]) -> Tuple[set[str], int]:
    all_ids: set[str] = set()
    total_records = 0

    for fasta_path in fasta_files:
        current_ids: set[str] = set()
        n_records = 0

        try:
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                record_id = str(record.id).strip()
                if not record_id:
                    raise ValueError(f"Blank FASTA record ID in: {fasta_path}")
                if record_id in current_ids:
                    raise ValueError(
                        f"Duplicated FASTA record ID '{record_id}' within file: {fasta_path}"
                    )
                if len(record.seq) == 0:
                    raise ValueError(
                        f"FASTA record '{record_id}' has an empty sequence: {fasta_path}"
                    )

                current_ids.add(record_id)
                all_ids.add(record_id)
                n_records += 1
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Failed to parse FASTA file: {fasta_path}\n{exc}") from exc

        if n_records == 0:
            raise ValueError(f"No FASTA records found in: {fasta_path}")

        print(f"[OK] FASTA: {fasta_path} ({n_records} records)")
        total_records += n_records

    return all_ids, total_records


def check_output_directory(base_dir: Path) -> None:
    if base_dir.exists() and not base_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {base_dir}")

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CREATE] Output directory: {base_dir}")
    else:
        print(f"[OK] Output directory: {base_dir}")

    try:
        with tempfile.NamedTemporaryFile(dir=base_dir, prefix=".write_test_", delete=True):
            pass
    except OSError as exc:
        raise PermissionError(f"Output directory is not writable: {base_dir}") from exc


def run_preflight_checks(config: PipelineConfig) -> None:
    print("=" * 68)
    print("[STEP 0] Preflight checks")
    print("=" * 68)
    print(f"[INFO] Working directory: {Path.cwd()}")

    # Input paths
    if not config.input_fasta_path.exists():
        raise FileNotFoundError(
            f"Input FASTA path not found: {config.input_fasta_path}"
        )

    if not config.feature_table_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {config.feature_table_path}"
        )
    if not config.feature_table_path.is_file():
        raise ValueError(
            f"Feature table path is not a file: {config.feature_table_path}"
        )

    fasta_files = list_fasta_files(config.input_fasta_path)
    if not fasta_files:
        raise FileNotFoundError(
            f"No FASTA input files found from: {config.input_fasta_path}"
        )

    fasta_ids, total_fasta_records = validate_fasta_inputs(fasta_files)
    feature_table, abundance_columns = read_feature_table(config.feature_table_path)

    table_ids = set(feature_table["ASV"].astype(str))
    overlap = fasta_ids & table_ids
    only_table = table_ids - fasta_ids
    only_fasta = fasta_ids - table_ids

    if not overlap:
        raise ValueError(
            "No ASV IDs overlap between FASTA input and abundance table. "
            "Please check whether the two inputs belong to the same dataset."
        )

    print(
        f"[OK] ASV ID overlap: {len(overlap)} IDs "
        f"(FASTA records: {total_fasta_records}; abundance rows: {len(table_ids)})"
    )
    if only_table:
        print(
            f"[WARN] {len(only_table)} ASV(s) occur in the abundance table but not in FASTA. "
            "They will remain in tables but cannot be exported as sequences."
        )
    if only_fasta:
        print(
            f"[INFO] {len(only_fasta)} ASV(s) occur in FASTA but not in the abundance table. "
            "They will not enter the final abundance-linked master table."
        )

    print(f"[OK] Feature table: {config.feature_table_path}")
    print(f"[OK] Number of abundance columns: {len(abundance_columns)}")

    # Output path
    check_output_directory(config.base_dir)

    # External tools
    blastn_executable = check_external_tool(config.blastn_bin, "blastn")
    taxonkit_executable = check_external_tool(config.taxonkit_bin, "taxonkit")
    check_external_tool(config.csvtk_bin, "csvtk")

    # Databases
    validate_blast_database(config.blast_db, blastn_executable)
    check_taxonkit_database(taxonkit_executable)

    print(f"[OK] Threads: {config.num_threads}")
    print("=" * 68)
    print("[PASS] All required preflight checks passed.")
    print("=" * 68)


# ============================================================
# Step 1: Run BLASTN top1
# ============================================================

def run_blast_top1(config: PipelineConfig) -> None:
    print("\n[STEP 1] Run BLASTN top1")
    ensure_dir(config.blast_dir)

    fasta_files = list_fasta_files(config.input_fasta_path)

    for fasta_path in fasta_files:
        sample_name = sample_name_from_fasta(fasta_path)
        output_path = config.blast_dir / f"{sample_name}_blast.txt"

        cmd = [
            config.blastn_bin,
            "-db", str(config.blast_db),
            "-query", str(fasta_path),
            "-outfmt", "6 " + " ".join(BLAST_COLUMNS),
            "-max_hsps", "1",
            "-max_target_seqs", "1",
            "-num_threads", str(config.num_threads),
            "-out", str(output_path),
        ]

        print(f"[RUN] {fasta_path.name} -> {output_path.name}")
        run_command(cmd)

    print(f"[DONE] BLAST outputs saved to: {config.blast_dir}")


# ============================================================
# Step 2: Orient sequences by BLAST direction
# ============================================================

def read_blast_top1(blast_path: Path) -> pd.DataFrame:
    if not blast_path.exists():
        raise FileNotFoundError(f"BLAST output not found: {blast_path}")

    if blast_path.stat().st_size == 0:
        return pd.DataFrame(columns=BLAST_COLUMNS)

    try:
        df = pd.read_csv(
            blast_path,
            sep="\t",
            header=None,
            names=BLAST_COLUMNS,
            dtype={"qseqid": str, "sseqid": str, "staxids": str},
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BLAST_COLUMNS)

    numeric_cols = [
        "pident", "qcovhsp", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def write_oriented_fasta(
    input_fasta_path: Path,
    blast_path: Path,
    output_fasta_path: Path,
) -> None:
    seq_dict: Dict[str, SeqRecord] = SeqIO.to_dict(
        SeqIO.parse(str(input_fasta_path), "fasta")
    )
    blast_df = read_blast_top1(blast_path)

    if blast_df.empty:
        print(
            f"[WARN] BLAST file is empty; keeping original orientation for all records: "
            f"{blast_path.name}"
        )
        records = [
            SeqRecord(record.seq, id=record.id, description="")
            for record in seq_dict.values()
        ]
        SeqIO.write(records, str(output_fasta_path), "fasta")
        return

    blast_df = blast_df.drop_duplicates(subset=["qseqid"], keep="first").copy()
    blast_df["is_forward"] = blast_df["sstart"] < blast_df["send"]
    orientation_map = blast_df.set_index("qseqid")["is_forward"].to_dict()

    oriented_records: List[SeqRecord] = []
    no_blast_hit: List[str] = []

    for record_id, record in seq_dict.items():
        is_forward = orientation_map.get(record_id)

        if is_forward is None:
            no_blast_hit.append(record_id)
            new_record = SeqRecord(record.seq, id=record.id, description="")
        elif bool(is_forward):
            new_record = SeqRecord(record.seq, id=record.id, description="")
        else:
            new_record = SeqRecord(
                record.seq.reverse_complement(),
                id=record.id,
                description="",
            )

        oriented_records.append(new_record)

    if no_blast_hit:
        print(
            f"[WARN] {len(no_blast_hit)} FASTA record(s) have no BLAST hit in "
            f"{blast_path.name}; original orientation retained."
        )

    SeqIO.write(oriented_records, str(output_fasta_path), "fasta")


def orient_sequences(config: PipelineConfig) -> None:
    print("\n[STEP 2] Orient FASTA sequences")
    ensure_dir(config.oriented_dir)

    fasta_files = list_fasta_files(config.input_fasta_path)

    for fasta_path in fasta_files:
        sample_name = sample_name_from_fasta(fasta_path)
        blast_path = config.blast_dir / f"{sample_name}_blast.txt"
        output_fasta_path = config.oriented_dir / f"{sample_name}_oriented.fasta"

        if not blast_path.exists():
            raise FileNotFoundError(
                f"BLAST result not found for orientation step: {blast_path}"
            )

        write_oriented_fasta(fasta_path, blast_path, output_fasta_path)
        print(f"[OK] {fasta_path.name} -> {output_fasta_path.name}")

    print(f"[DONE] Oriented FASTA saved to: {config.oriented_dir}")


# ============================================================
# Step 3: Extract taxonomy with TaxonKit
# ============================================================

def extract_taxids_from_blast(blast_path: Path, taxid_path: Path) -> None:
    blast_df = read_blast_top1(blast_path)

    if blast_df.empty:
        taxid_path.write_text("", encoding="utf-8")
        return

    # staxids may contain multiple IDs separated by ';'. TaxonKit needs one taxid
    # per lookup row, so use the first taxid while retaining the original staxids
    # field in the BLAST table.
    taxids = (
        blast_df["staxids"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .str[0]
        .str.strip()
    )
    taxids = taxids[taxids != ""]

    taxids.to_csv(
        taxid_path,
        sep="\t",
        index=False,
        header=False,
    )


def build_taxonomy_table(
    config: PipelineConfig,
    taxid_path: Path,
    taxonomy_path: Path,
) -> None:
    if not taxid_path.exists() or taxid_path.stat().st_size == 0:
        pd.DataFrame(columns=TAXON_COLUMNS).to_csv(
            taxonomy_path,
            sep="\t",
            index=False,
        )
        return

    lineage_path = taxonomy_path.with_suffix(".lineage.tmp.txt")

    try:
        run_command(
            [config.taxonkit_bin, "lineage", str(taxid_path)],
            stdout_path=lineage_path,
        )

        taxonkit_q = shlex.quote(config.taxonkit_bin)
        csvtk_q = shlex.quote(config.csvtk_bin)
        lineage_q = shlex.quote(str(lineage_path))

        cmd = (
            f"{taxonkit_q} reformat {lineage_q} | "
            f"{csvtk_q} -H -t cut -f 1,3 | "
            f"{csvtk_q} -H -t sep -f 2 -s ';' -R | "
            f"{csvtk_q} add-header -t "
            f"-n taxid,kingdom,phylum,class,order,Family,Genus,Species"
        )
        run_command(cmd, stdout_path=taxonomy_path, shell=True)
    finally:
        if lineage_path.exists():
            lineage_path.unlink()


def combine_blast_and_taxonomy(
    blast_path: Path,
    taxonomy_path: Path,
    output_path: Path,
) -> None:
    blast_df = read_blast_top1(blast_path)

    if blast_df.empty:
        pd.DataFrame(columns=FINAL_TAXON_COLUMNS).to_csv(
            output_path,
            sep="\t",
            index=False,
        )
        print(f"[WARN] Empty BLAST table: {blast_path.name}")
        return

    try:
        tax_df = pd.read_csv(taxonomy_path, sep="\t", dtype={"taxid": str})
    except pd.errors.EmptyDataError:
        tax_df = pd.DataFrame(columns=TAXON_COLUMNS)

    if tax_df.empty:
        tax_df = pd.DataFrame(columns=TAXON_COLUMNS)

    for col in TAXON_COLUMNS:
        if col not in tax_df.columns:
            tax_df[col] = pd.NA

    tax_df["taxid"] = tax_df["taxid"].astype(str)
    tax_df = tax_df.drop_duplicates(subset=["taxid"], keep="first")

    blast_df = blast_df.copy()
    blast_df["taxid_lookup"] = (
        blast_df["staxids"]
        .fillna("")
        .astype(str)
        .str.split(";")
        .str[0]
        .str.strip()
    )

    combined = blast_df.merge(
        tax_df,
        left_on="taxid_lookup",
        right_on="taxid",
        how="left",
    )
    combined = combined.drop(columns=["taxid_lookup"], errors="ignore")

    combined["pident"] = pd.to_numeric(combined["pident"], errors="coerce")
    combined["qcovhsp"] = pd.to_numeric(combined["qcovhsp"], errors="coerce")

    # Retain taxonomy only when both BLAST metrics pass the fixed thresholds
    # and both Genus and Species are available.
    low_confidence = (
        combined["pident"].isna()
        | combined["qcovhsp"].isna()
        | (combined["pident"] < PIDENT_THRESHOLD)
        | (combined["qcovhsp"] < QCOVHSP_THRESHOLD)
        | combined["Genus"].isna()
        | combined["Species"].isna()
    )

    n_unassigned = int(low_confidence.sum())
    combined.loc[low_confidence, ["Genus", "Species"]] = "unassigned"

    for col in ["Genus", "Species"]:
        combined[col] = combined[col].fillna("unassigned").map(clean_parentheses)

    if n_unassigned > 0:
        print(
            f"[INFO] {n_unassigned}/{len(combined)} taxonomy assignment(s) "
            "set to unassigned."
        )

    combined = combined[FINAL_TAXON_COLUMNS]
    combined.to_csv(output_path, sep="\t", index=False)


def process_blast_taxonomy(config: PipelineConfig) -> None:
    print("\n[STEP 3] Extract taxonomy from BLAST taxids")
    ensure_dir(config.taxon_dir)

    blast_files = sorted(config.blast_dir.glob("*_blast.txt"))
    if not blast_files:
        raise FileNotFoundError(f"No BLAST files found in: {config.blast_dir}")

    tmp_dir = config.taxon_dir / "tmp"
    ensure_dir(tmp_dir)

    for blast_path in blast_files:
        sample_name = blast_path.name.replace("_blast.txt", "")
        taxid_path = tmp_dir / f"{sample_name}_taxids.txt"
        taxonomy_path = tmp_dir / f"{sample_name}_taxonomy.txt"
        output_path = config.taxon_dir / f"{sample_name}_blast_combined.txt"

        extract_taxids_from_blast(blast_path, taxid_path)
        build_taxonomy_table(config, taxid_path, taxonomy_path)
        combine_blast_and_taxonomy(blast_path, taxonomy_path, output_path)

        taxid_path.unlink(missing_ok=True)
        taxonomy_path.unlink(missing_ok=True)

        print(f"[OK] {blast_path.name} -> {output_path.name}")

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"[DONE] Taxonomy outputs saved to: {config.taxon_dir}")

# ============================================================
# Step 4: Build checked ASV taxonomy and sequence maps
# ============================================================

def find_conflicts(
    df: pd.DataFrame,
    group_col: str,
    compare_cols: Sequence[str],
) -> pd.DataFrame:
    """Return conflicting duplicated rows without writing an output file."""
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    existing_compare_cols = [c for c in compare_cols if c in df.columns]
    if not existing_compare_cols:
        return pd.DataFrame(columns=df.columns)

    nunique = df.groupby(group_col)[existing_compare_cols].nunique(dropna=False)
    conflict_ids = nunique.index[(nunique > 1).any(axis=1)].tolist()

    if not conflict_ids:
        return pd.DataFrame(columns=df.columns)

    conflicts = df[df[group_col].isin(conflict_ids)].copy()

    sort_cols = [group_col]
    for col in ["source_sample", "source_file"]:
        if col in conflicts.columns:
            sort_cols.append(col)

    return conflicts.sort_values(sort_cols)


def build_checked_asv_taxonomy_map(config: PipelineConfig) -> pd.DataFrame:
    print("\n[STEP 4.1] Build checked ASV taxonomy map")
    ensure_dir(config.abundance_dir)

    tax_files = sorted(config.taxon_dir.glob("*_blast_combined.txt"))
    if not tax_files:
        raise FileNotFoundError(f"No taxonomy files found in: {config.taxon_dir}")

    tax_dfs: List[pd.DataFrame] = []

    for path in tax_files:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                dtype={"qseqid": str, "sseqid": str, "staxids": str, "taxid": str},
            )
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=FINAL_TAXON_COLUMNS)

        if df.empty:
            continue

        required = {
            "qseqid", "sseqid", "pident", "qcovhsp", "evalue", "bitscore",
            "staxids", "Genus", "Species",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

        sample_name = sample_name_from_taxonomy(path)

        keep_cols = [
            "qseqid", "sseqid", "pident", "qcovhsp", "evalue", "bitscore",
            "staxids", "taxid", "kingdom", "phylum", "class", "order",
            "Family", "Genus", "Species",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]

        tmp = df[keep_cols].copy()
        tmp = tmp.rename(columns={"qseqid": "ASV"})
        tmp["source_sample"] = sample_name
        tmp["source_file"] = path.name

        tmp["Genus"] = tmp["Genus"].fillna("unassigned").map(clean_parentheses)
        tmp["Species"] = tmp["Species"].fillna("unassigned").map(clean_parentheses)

        for col in ["pident", "qcovhsp", "evalue", "bitscore"]:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        tax_dfs.append(tmp)

    if not tax_dfs:
        columns = [
            "ASV", "sseqid", "pident", "qcovhsp", "evalue", "bitscore",
            "staxids", "taxid", "kingdom", "phylum", "class", "order", "Family",
            "Genus", "Species",
            "taxonomy_source_samples", "taxonomy_source_files", "n_taxonomy_source_rows",
        ]
        representative = pd.DataFrame(columns=columns)
        print("[WARN] No ASVs had BLAST taxonomy annotations.")
        return representative

    all_tax = pd.concat(tax_dfs, ignore_index=True)

    duplicate_extra = int(all_tax["ASV"].duplicated(keep="first").sum())
    if duplicate_extra > 0:
        print(
            f"[INFO] {duplicate_extra} duplicated ASV taxonomy row(s) found across "
            "sample-level BLAST files. This is expected when the same global ASV "
            "appears in multiple samples."
        )

    compare_cols = [
        "sseqid", "staxids", "Genus", "Species",
        "pident", "qcovhsp", "evalue", "bitscore",
    ]
    conflicts = find_conflicts(all_tax, "ASV", compare_cols)

    if conflicts.empty:
        print("[OK] No duplicated-ASV taxonomy conflicts.")
    else:
        print(
            f"[WARN] Duplicated-ASV taxonomy conflicts found for "
            f"{conflicts['ASV'].nunique()} ASV(s). "
            "The highest-ranked annotation will be used."
        )

    # If the same global ASV occurs in several sample-level files, choose one
    # representative annotation deterministically.
    tmp = all_tax.copy()
    tmp["_bitscore_sort"] = tmp["bitscore"].fillna(float("-inf"))
    tmp["_qcovhsp_sort"] = tmp["qcovhsp"].fillna(float("-inf"))
    tmp["_pident_sort"] = tmp["pident"].fillna(float("-inf"))
    tmp["_evalue_sort"] = tmp["evalue"].fillna(float("inf"))

    representative = tmp.sort_values(
        by=[
            "ASV",
            "_bitscore_sort",
            "_qcovhsp_sort",
            "_pident_sort",
            "_evalue_sort",
            "source_file",
        ],
        ascending=[True, False, False, False, True, True],
    ).drop_duplicates(subset=["ASV"], keep="first")

    source_summary = all_tax.groupby("ASV").agg(
        taxonomy_source_samples=(
            "source_sample",
            lambda x: ";".join(sorted(set(map(str, x)))),
        ),
        taxonomy_source_files=(
            "source_file",
            lambda x: ";".join(sorted(set(map(str, x)))),
        ),
        n_taxonomy_source_rows=("source_file", "size"),
    ).reset_index()

    representative = representative.drop(
        columns=["_bitscore_sort", "_qcovhsp_sort", "_pident_sort", "_evalue_sort"],
        errors="ignore",
    )
    representative = representative.merge(source_summary, on="ASV", how="left")

    print(f"[DONE] Checked taxonomy map built in memory: {len(representative)} ASV(s).")
    return representative


def build_checked_asv_sequence_map(config: PipelineConfig) -> pd.DataFrame:
    print("\n[STEP 4.2] Build checked ASV sequence map")
    ensure_dir(config.abundance_dir)

    oriented_files = sorted(config.oriented_dir.glob("*_oriented.fasta"))
    if not oriented_files:
        raise FileNotFoundError(f"No oriented FASTA files found in: {config.oriented_dir}")

    rows: List[dict] = []

    for path in oriented_files:
        source_sample = sample_name_from_oriented_fasta(path)
        for record in SeqIO.parse(str(path), "fasta"):
            rows.append({
                "ASV": str(record.id),
                "sequence": str(record.seq).upper(),
                "sequence_source_sample": source_sample,
                "sequence_source_file": path.name,
            })

    seq_df = pd.DataFrame(rows)
    if seq_df.empty:
        raise ValueError("No records found in oriented FASTA files.")

    duplicate_extra = int(seq_df["ASV"].duplicated(keep="first").sum())
    if duplicate_extra > 0:
        print(
            f"[INFO] {duplicate_extra} duplicated ASV sequence row(s) found across "
            "sample-level FASTA files."
        )

    conflicts = find_conflicts(seq_df, "ASV", ["sequence"])

    if conflicts.empty:
        print("[OK] No duplicated-ASV sequence conflicts.")
    else:
        print(
            f"[WARN] Duplicated-ASV sequence conflicts found for "
            f"{conflicts['ASV'].nunique()} ASV(s). "
            "A deterministic representative sequence will be used."
        )

    representative = seq_df.sort_values(
        by=["ASV", "sequence_source_file"],
        ascending=[True, True],
    ).drop_duplicates(subset=["ASV"], keep="first")

    source_summary = seq_df.groupby("ASV").agg(
        sequence_source_samples=(
            "sequence_source_sample",
            lambda x: ";".join(sorted(set(map(str, x)))),
        ),
        sequence_source_files=(
            "sequence_source_file",
            lambda x: ";".join(sorted(set(map(str, x)))),
        ),
        n_sequence_source_rows=("sequence_source_file", "size"),
    ).reset_index()

    representative = representative.drop(
        columns=["sequence_source_sample", "sequence_source_file"],
        errors="ignore",
    )
    representative = representative.merge(source_summary, on="ASV", how="left")

    print(f"[DONE] Checked sequence map built in memory: {len(representative)} ASV(s).")
    return representative


# ============================================================
# Step 5: Build master ASV table and assign FASTA IDs
# ============================================================

def build_master_asv_table(
    config: PipelineConfig,
    feature_table: pd.DataFrame,
    abundance_columns: Sequence[str],
    tax_map: pd.DataFrame,
    seq_map: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[STEP 5.1] Build master ASV table")
    ensure_dir(config.abundance_dir)

    master = feature_table.merge(tax_map, on="ASV", how="left")
    master = master.merge(seq_map, on="ASV", how="left")

    missing_tax = (
        int(master["Species"].isna().sum())
        if "Species" in master.columns
        else len(master)
    )
    missing_seq = (
        int(master["sequence"].isna().sum())
        if "sequence" in master.columns
        else len(master)
    )

    if missing_tax > 0:
        print(
            f"[WARN] {missing_tax} ASV(s) have no taxonomy annotation after merge; "
            "Genus/Species set to unassigned."
        )
    if missing_seq > 0:
        print(
            f"[WARN] {missing_seq} ASV(s) have no oriented sequence after merge; "
            "they remain in tables but cannot be exported to FASTA."
        )

    for col in ["Genus", "Species"]:
        if col not in master.columns:
            master[col] = "unassigned"
        master[col] = master[col].fillna("unassigned").map(clean_parentheses)


    master["Total"] = master[list(abundance_columns)].sum(axis=1)
    total_sum = master["Total"].sum()

    if total_sum == 0:
        master["Abundance Ratio"] = 0.0
    else:
        master["Abundance Ratio"] = (master["Total"] / total_sum * 100).round(6)

    master["Species_clean"] = master["Species"].map(clean_taxon_name)
    master["Genus_clean"] = master["Genus"].map(clean_taxon_name)

    # A named species is eligible as a target candidate. Unassigned sequences are
    # nevertheless retained as background for downstream off-target screening.
    master["is_target_candidate"] = master["Species_clean"] != "unassigned"
    master["is_background_database"] = (
        master["sequence"].notna()
        & (master["sequence"].astype(str).str.len() > 0)
    )

    master["ASV_numeric"] = master["ASV"].map(asv_numeric_id)
    master = master.sort_values(
        by=["Total", "ASV_numeric"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return master


def assign_final_fasta_ids(
    config: PipelineConfig,
    master_df: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[STEP 5.2] Assign final FASTA record IDs")

    df = master_df.copy()
    df["final_fasta_record_id"] = pd.NA

    eligible_mask = (
        df["sequence"].notna()
        & (df["sequence"].astype(str).str.len() > 0)
    )

    if not config.export_unassigned_fasta:
        eligible_mask = eligible_mask & (df["Species_clean"] != "unassigned")

    eligible = df[eligible_mask].copy()

    n_records = 0
    n_taxa = 0

    for species_clean, group in eligible.groupby("Species_clean", sort=True):
        group_sorted = group.sort_values(
            by=["Total", "ASV_numeric"],
            ascending=[False, True],
        )

        for i, idx in enumerate(group_sorted.index, start=1):
            df.at[idx, "final_fasta_record_id"] = f"{species_clean}_{i}"
            n_records += 1

        n_taxa += 1

    print(f"[DONE] Assigned FASTA IDs for {n_records} ASV(s) in {n_taxa} group(s).")
    return df


# ============================================================
# Step 6: Export tables and FASTA files
# ============================================================

def summarize_abundance(
    master_df: pd.DataFrame,
    group_col: str,
    abundance_columns: Sequence[str],
) -> pd.DataFrame:
    summary = master_df.groupby(group_col, dropna=False).agg(
        n_ASV=("ASV", "nunique"),
        Total=("Total", "sum"),
        **{col: (col, "sum") for col in abundance_columns},
    ).reset_index()

    total_sum = summary["Total"].sum()
    if total_sum == 0:
        summary["Abundance Ratio"] = 0.0
    else:
        summary["Abundance Ratio"] = (summary["Total"] / total_sum * 100).round(6)

    return summary.sort_values(by="Total", ascending=False).reset_index(drop=True)


def export_summary_tables(
    config: PipelineConfig,
    master_df: pd.DataFrame,
    abundance_columns: Sequence[str],
) -> None:
    print("\n[STEP 6.1] Export abundance summary tables")
    ensure_dir(config.abundance_dir)

    species_summary = summarize_abundance(
        master_df,
        "Species",
        abundance_columns,
    )
    species_summary_path = config.abundance_dir / "Species_summary_abundance.csv"
    species_summary.to_csv(species_summary_path, index=False)

    genus_summary = summarize_abundance(
        master_df,
        "Genus",
        abundance_columns,
    )
    genus_summary_path = config.abundance_dir / "Genus_summary_abundance.csv"
    genus_summary.to_csv(genus_summary_path, index=False)

    print(f"[DONE] Species summary: {species_summary_path}")
    print(f"[DONE] Genus summary: {genus_summary_path}")


def export_global_fasta_from_master(
    config: PipelineConfig,
    master_df: pd.DataFrame,
) -> None:
    print("\n[STEP 6.2] Export global FASTA database")
    ensure_dir(config.species_fasta_dir)

    fasta_df = master_df[
        master_df["final_fasta_record_id"].notna()
        & master_df["sequence"].notna()
        & (master_df["sequence"].astype(str).str.len() > 0)
    ].copy()

    if fasta_df.empty:
        raise ValueError("No FASTA-eligible ASVs found in master table.")

    fasta_df = fasta_df.sort_values(
        by=["Species_clean", "Total", "ASV_numeric"],
        ascending=[True, False, True],
    )

    global_output = config.species_fasta_dir / "global_all_taxa_consensus.fasta"

    with open(global_output, "w", encoding="utf-8") as fout:
        for _, row in fasta_df.iterrows():
            record_id = str(row["final_fasta_record_id"])
            seq = str(row["sequence"]).upper()
            fout.write(f">{record_id}\n")
            fout.write(f"{wrap_sequence(seq)}\n")

    n_unassigned = int((fasta_df["Species_clean"] == "unassigned").sum())

    print(f"[DONE] FASTA records in global database: {len(fasta_df)}")
    print(f"[DONE] Unassigned records included for off-target screening: {n_unassigned}")
    print(f"[DONE] Global FASTA saved to: {global_output}")


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PacBio 16S ASV processing pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.json"),
        help=(
            "Path to params.json. Default: params.json. "
            "Relative paths inside params.json are interpreted from the current "
            "working directory."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run preflight checks only; do not execute the analysis pipeline.",
    )
    return parser.parse_args()


def main(config: PipelineConfig, check_only: bool = False) -> None:
    run_preflight_checks(config)

    if check_only:
        print("\n[DONE] Preflight checks completed; analysis was not started (--check-only).")
        return

    run_blast_top1(config)
    orient_sequences(config)
    process_blast_taxonomy(config)

    feature_table, abundance_columns = read_feature_table(config.feature_table_path)
    tax_map = build_checked_asv_taxonomy_map(config)
    seq_map = build_checked_asv_sequence_map(config)

    master_df = build_master_asv_table(
        config=config,
        feature_table=feature_table,
        abundance_columns=abundance_columns,
        tax_map=tax_map,
        seq_map=seq_map,
    )

    master_df = assign_final_fasta_ids(config, master_df)

    export_summary_tables(
        config=config,
        master_df=master_df,
        abundance_columns=abundance_columns,
    )
    export_global_fasta_from_master(config, master_df)

    print("\n[DONE] Processing completed successfully.")


if __name__ == "__main__":
    try:
        args = parse_args()
        CONFIG = load_config(args.config)
        main(CONFIG, check_only=args.check_only)
    except (FileNotFoundError, PermissionError, ValueError, KeyError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
