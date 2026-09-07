# coding=utf-8
import pandas as pd


def check_ships_file(ships_df):
    """
    Audit SHIPS wind dataframe for duplicate/conflicting records.

    Expected:
        - ships_df.index = timestamp
        - ships_df.atcf_id = storm ID
        - ships_df.max_winds = intensity value
    """

    df = ships_df.copy()

    # Normalize time
    df.index = pd.to_datetime(df.index)

    print("===== SHIPS FILE CHECK =====")
    print(f"Total records: {len(df):,}")
    print(f"Unique storms: {df['atcf_id'].nunique():,}")
    print()

    # --------------------------------------------------
    # Duplicate storm/time combinations
    # --------------------------------------------------
    duplicates = (
        df.groupby(["atcf_id", df.index])
        .size()
        .reset_index(name="count")
    )

    duplicates = duplicates[duplicates["count"] > 1]

    print(f"Duplicate storm/time entries: {len(duplicates):,}")

    if len(duplicates) > 0:
        print("\nExamples:")
        print(duplicates.head(20))

    print()

    # --------------------------------------------------
    # Conflicting winds
    # --------------------------------------------------
    conflicts = (
        df.groupby(["atcf_id", df.index])["max_winds"]
        .nunique()
        .reset_index(name="unique_winds")
    )

    conflicts = conflicts[conflicts["unique_winds"] > 1]

    print(f"Conflicting wind values: {len(conflicts):,}")

    if len(conflicts) > 0:
        print("\nConflicts:")

        conflict_idx = conflicts[
            ["atcf_id", "level_1"]
        ].values.tolist()

        for storm, time in conflict_idx[:10]:
            subset = df[
                (df["atcf_id"] == storm) &
                (df.index == time)
                ]

            print("\n", storm, time)
            print(subset[["max_winds"]])

    print()

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------
    print("Missing max winds:")
    print(df["max_winds"].isna().sum())

    print()

    # --------------------------------------------------
    # Timestamp summary
    # --------------------------------------------------
    hours = pd.Series(df.index.hour).value_counts().sort_index()

    print("Timestamp hour distribution:")
    print(hours)

    return duplicates, conflicts

wind_dict = pd.read_csv('//uahdata/rstor/cataloging/nc_process/violin_plots/'
                        'shear_process_all/ships_interp.txt', sep='\t', index_col=0)
duplicates, conflicts = check_ships_file(wind_dict)
print(duplicates)
print(conflicts)