# PacBio 16S rRNA Reference Database Construction

This module processes PacBio full-length 16S rRNA ASV sequences to generate a taxonomy-annotated, abundance-linked sequence database for downstream probe design.

The workflow performs BLAST-based taxonomic annotation, sequence orientation, taxonomy retrieval with TaxonKit, abundance integration, and export of genus/species abundance summaries together with a merged 16S FASTA database.

---

## Workflow

The pipeline performs the following steps:

1. **Preflight checks**
   - Validate input FASTA and abundance table
   - Check ASV ID consistency
   - Check the BLAST nucleotide database
   - Check required external tools
   - Verify that the TaxonKit taxonomy database is functional

2. **BLASTN annotation**
   - Run BLASTN against a local NCBI 16S rRNA database
   - Retain the top BLAST hit for each ASV

3. **Sequence orientation**
   - Orient each ASV sequence according to the BLAST alignment direction

4. **Taxonomy assignment**
   - Retrieve taxonomic lineage information from BLAST taxids using TaxonKit
   - Apply fixed sequence-identity and query-coverage thresholds
   - Assign low-confidence records as `unassigned`

5. **ASV integration**
   - Combine taxonomy, abundance, and oriented sequences at the ASV level
   - Assign standardized FASTA record IDs

6. **Final output**
   - Generate species-level abundance summary
   - Generate genus-level abundance summary
   - Generate a merged global 16S FASTA database

Unassigned ASVs are retained in the final FASTA database because they remain useful as background sequences for downstream off-target screening.

---

## Input files

The example dataset contains two input files:

```text
example/
├── example_input.fasta
└── example_asv_abundance.csv
```

### FASTA file

The FASTA file contains full-length 16S ASV sequences.

Example:

```text
>ASV_1
GATGAACGCTAGCGGCAGGCTTAACACATGCA...
>ASV_2
AGAGTTTGATCCTGGCTCAG...
```

FASTA record IDs must correspond to the ASV IDs in the abundance table.

### Abundance table

The abundance table must contain an `ASV` column followed by one or more abundance columns.

Example:

```csv
ASV,example
ASV_1,4328
ASV_2,2622
ASV_3,1584
```

All columns other than `ASV` are automatically treated as abundance columns.

For datasets with multiple samples:

```csv
ASV,sample_1,sample_2,sample_3
ASV_1,4328,3910,4215
ASV_2,2622,2501,2810
```

---

## Dependencies

### Python

Tested with:

- Python 3.12.2
- pandas 2.3.1
- Biopython 1.84

Install the required Python packages with:

```bash
pip install pandas biopython
```

### External command-line tools

Tested with:

- NCBI BLAST+ 2.16.0+
- TaxonKit v0.17.0
- csvtk v0.30.0

The following commands must be available in `PATH`:

- `blastn`
- `blastdbcmd` (used for database validation when available)
- `taxonkit`
- `csvtk`

Other compatible versions may also work.

The pipeline checks these dependencies before starting the analysis.

---

## Database setup

This pipeline uses three external database resources:

1. NCBI `16S_ribosomal_RNA` BLAST database
2. BLAST taxonomy support files (`taxdb`)
3. NCBI taxonomy dump used by TaxonKit (`taxdump`)

These database files are not included in the repository and should be downloaded from their official sources.

### 1. NCBI 16S rRNA BLAST database

The preformatted NCBI `16S_ribosomal_RNA` database is used by `blastn`.

NCBI provides the `update_blastdb.pl` utility with BLAST+ for downloading preformatted databases.

Recommended download command:

```bash
mkdir -p db
cd db

update_blastdb.pl --decompress 16S_ribosomal_RNA
```

Depending on the BLAST+ installation, the script may need to be called through Perl:

```bash
perl /path/to/update_blastdb.pl --decompress 16S_ribosomal_RNA
```

Official NCBI BLAST database directory:

https://ftp.ncbi.nlm.nih.gov/blast/db/

NCBI BLAST database documentation:

https://www.ncbi.nlm.nih.gov/books/NBK52640/

The pipeline configuration uses the **database prefix**, not an individual database file:

```json
"blast_db": "db/16S_ribosomal_RNA"
```

### 2. BLAST taxonomy support files

BLAST taxonomy support files are distributed by NCBI as `taxdb.tar.gz`.

Recommended download command:

```bash
cd db

update_blastdb.pl --decompress taxdb
```

Alternatively, the archive can be downloaded directly from:

https://ftp.ncbi.nlm.nih.gov/blast/db/taxdb.tar.gz

After decompression, the relevant files include:

```text
taxdb.btd
taxdb.bti
taxonomy4blast.sqlite3
```

These files should normally be placed in the same directory as the BLAST database.

The BLAST taxonomy files are used by BLAST/`blastdbcmd` and are separate from the taxonomy database used by TaxonKit.

### 3. TaxonKit taxonomy database

TaxonKit requires the NCBI taxonomy dump (`taxdump.tar.gz`).

Download the official NCBI taxonomy archive:

```bash
wget -c https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar -zxvf taxdump.tar.gz
```

TaxonKit uses the following files:

```text
names.dmp
nodes.dmp
delnodes.dmp
merged.dmp
```

By default, TaxonKit expects these files under:

```text
$HOME/.taxonkit/
```

Install them with:

```bash
mkdir -p $HOME/.taxonkit
cp names.dmp nodes.dmp delnodes.dmp merged.dmp $HOME/.taxonkit/
```

Official NCBI taxonomy archive:

https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz

TaxonKit documentation:

https://bioinf.shenwei.me/taxonkit/

A custom TaxonKit database directory can also be specified using TaxonKit's `--data-dir` option or the `TAXONKIT_DB` environment variable.

---

## Configuration

The pipeline is controlled by a JSON configuration file.

Example `config/params.json`:

```json
{
  "base_dir": "output",
  "input_fasta_path": "example/example_input.fasta",
  "feature_table_path": "example/example_asv_abundance.csv",
  "blast_db": "db/16S_ribosomal_RNA",
  "blastn_bin": "blastn",
  "taxonkit_bin": "taxonkit",
  "csvtk_bin": "csvtk",
  "num_threads": 8,
  "export_unassigned_fasta": true
}
```

### Configuration fields

| Parameter | Description |
|---|---|
| `base_dir` | Output directory |
| `input_fasta_path` | Input ASV FASTA file |
| `feature_table_path` | ASV abundance table |
| `blast_db` | Prefix of the local 16S BLAST database |
| `blastn_bin` | BLASTN executable |
| `taxonkit_bin` | TaxonKit executable |
| `csvtk_bin` | csvtk executable |
| `num_threads` | Number of BLAST threads |
| `export_unassigned_fasta` | Whether unassigned ASVs are retained in the final FASTA database |

---

## Taxonomy filtering

The current pipeline uses fixed thresholds defined directly in the Python script:

```python
PIDENT_THRESHOLD = 85.0
QCOVHSP_THRESHOLD = 90.0
```

For each ASV, the top BLAST hit is first used to retrieve taxonomy.

If either:

- `pident < PIDENT_THRESHOLD`, or
- `qcovhsp < QCOVHSP_THRESHOLD`

the final `Genus` and `Species` assignments are set to:

```text
unassigned
```

Records with missing genus/species taxonomy are also treated as `unassigned`.

These sequences are not removed from the dataset and can still be retained in the merged FASTA database for off-target screening.

---

## Running the pipeline

Run commands from the module root directory.

### 1. Preflight check

It is recommended to first verify all inputs, databases, and dependencies without starting the analysis:

```bash
python pacbio_16S_pipeline.py \
    --config config/params.json \
    --check-only
```

A successful check ends with:

```text
[PASS] All required preflight checks passed.
```

### 2. Run the complete pipeline

```bash
python pacbio_16S_pipeline.py \
    --config config/params.json
```

---

## Output structure

Intermediate files are retained for troubleshooting and reproducibility.

```text
output/
├── blast/
├── oriented/
├── blast_tax/
├── table_abundance/
│   ├── Species_summary_abundance.csv
│   └── Genus_summary_abundance.csv
└── fin_clutser_sequence/
    └── global_all_taxa_consensus.fasta
```

### Intermediate directories

#### `blast/`

Contains top-hit BLASTN results for the input ASVs.

#### `oriented/`

Contains ASV sequences reoriented according to BLAST alignment direction.

#### `blast_tax/`

Contains BLAST results combined with TaxonKit-derived taxonomy.

### Final outputs

#### `Species_summary_abundance.csv`

Species-level ASV abundance summary.

#### `Genus_summary_abundance.csv`

Genus-level ASV abundance summary.

#### `global_all_taxa_consensus.fasta`

Merged 16S sequence database containing all FASTA-eligible ASVs.

FASTA IDs follow the format:

```text
>Species_name_number
```

For example:

```text
>Waltera_intestinalis_1
ATCG...
```

Unassigned ASVs are represented using IDs such as:

```text
>unassigned_1
```

---

## Example project structure

```text
pacbio_16S/
├── pacbio_16S_pipeline.py
├── README.md
├── config/
│   └── params.json
├── example/
│   ├── example_input.fasta
│   └── example_asv_abundance.csv
├── db/ # local database files, not included
└── output/ # generated files, not included
```
