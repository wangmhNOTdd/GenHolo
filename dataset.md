2024-06/
|-- v2
    |-- index # Consolidated tabular annotations
    |   |-- annotation_table.parquet
    |   |-- annotation_table_nonredundant.parquet
    |-- systems  # Structure files for all systems (split by `two_char_code` and zipped)
    |   |-- {two_char_code}.zip
    |-- clusters # Pre-calculated cluster labels derived from the protein similarity dataset
    |   |-- cluster=communities
    |       |-- ...
    |   |-- cluster=components
    |       |-- ...
    |-- splits # Split files and the configs used to generate them (if available)
    |   |-- split.parquet
    |   |-- split.yaml
    |-- linked_structures # Apo and predicted structures linked to their holo systems
    |   |-- {two_char_code}.zip
    |-- links # Apo and predicted structures similarity to their holo structures
    |   |-- apo_links.parquet
    |   |-- pred_links.parquet
    |
--------------------------------------------------------------------------------
                            miscellaneous data below
--------------------------------------------------------------------------------
    |
    |-- dbs # TSVs containing the raw files and IDs in the foldseek and mmseqs sub-databases
    |   |-- subdbs
    |       |-- apo.csv
    |       |-- holo.csv
    |       |-- pred.csv
    |-- entries # Raw annotations prior to consolidation (split by `two_char_code` and zipped)
    |   |-- {two_char_code}.zip
    |-- fingerprints # Index mapping files for the ligand similarity dataset
    |   |-- ligands_per_inchikey.parquet
    |   |-- ligands_per_inchikey_ecfp4.npy
    |   |-- ligands_per_system.parquet
    |-- ligand_scores # Ligand similarity parquet dataset
    |   |-- {hashid}.parquet
    |-- ligands # Ligand data expanded from entries for computing similarity
    |   |-- {hashid}.parquet
    |-- mmp # Ligand matched molecular pairs (MMP) and series (MMS) data
    |   |-- plinder_mmp_series.parquet
    |   |-- plinder_mms.csv.gz
    |-- scores # Protein similarity parquet dataset
    |   |-- search_db=apo
    |       |-- apo.parquet
    |   |-- search_db=holo
    |       |-- {chunck_id}.parquet
    |   |-- search_db=pred
    |       |-- pred.parquet