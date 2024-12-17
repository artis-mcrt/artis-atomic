import typing as t
from pathlib import Path

import pandas as pd
import polars as pl

import artisatomic

# from astropy import units as u
# import os.path
# import pandas as pd
# from carsus.util import parse_selected_species

BASEPATH = Path(Path.home() / "Google Drive/Shared drives/Atomic Data Group/FloersOpacityPaper/OutputFiles")


def extend_ion_list(ion_handlers, calibrated=True):
    assert BASEPATH.is_dir()
    handlername = "floers25" + "calib" if calibrated else "uncalib"
    for s in BASEPATH.glob("*_levels_calib.txt"):
        ionstr = s.name.lstrip("0123456789").split("_")[0]
        elsym = ionstr.rstrip("IVX")
        ion_stage_roman = ionstr.removeprefix(elsym)
        atomic_number = artisatomic.elsymbols.index(elsym)

        ion_stage = artisatomic.roman_numerals.index(ion_stage_roman)

        found_element = False
        for tmp_atomic_number, list_ions in ion_handlers:
            if tmp_atomic_number == atomic_number:
                if ion_stage not in [x[0] if len(x) > 0 else x for x in list_ions]:
                    list_ions.append((ion_stage, handlername))
                    list_ions.sort()
                found_element = True
        if not found_element:
            ion_handlers.append(
                (
                    atomic_number,
                    [(ion_stage, handlername)],
                )
            )

    ion_handlers.sort(key=lambda x: x[0])
    return ion_handlers


class FloersEnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: int


class FloersTransition(t.NamedTuple):
    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


def read_levels_and_transitions(atomic_number: int, ion_stage: int, flog, calibrated: bool):
    # ion_charge = ion_stage - 1
    elsym = artisatomic.elsymbols[atomic_number]
    ion_stage_roman = artisatomic.roman_numerals[ion_stage]
    calibstr = "calib" if calibrated else "uncalib"

    ionstr = f"{atomic_number}{elsym}{ion_stage_roman}"
    levels_file = BASEPATH / f"{ionstr}_levels_{calibstr}.txt"
    lines_file = BASEPATH / f"{ionstr}_transitions_{calibstr}.txt"

    artisatomic.log_and_print(
        flog,
        f"Reading Floers+25 {calibstr}rated data for Z={atomic_number} ion_stage {ion_stage} ({elsym} {ion_stage_roman})",
    )

    ionization_energy_in_ev = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]

    assert Path(levels_file).exists()
    dflevels = pl.from_pandas(
        pd.read_csv(levels_file, sep=r"\s+", skiprows=18, dtype_backend="pyarrow", dtype={"J": str})
    ).with_columns(
        pl.when(pl.col("J").str.ends_with("/2"))
        .then(pl.col("J").str.strip_suffix("/2").cast(pl.Int32) + 1)
        .otherwise(
            pl.col("J").str.strip_suffix("/2").cast(pl.Int32) * 2 + 1
        )  # the strip_suffix should not be needed (does not end in "/2" but prevents a polars error)
        .alias("g")
    )

    dflevels = dflevels.with_columns(pl.col("J").str.strip_suffix("/2").cast(pl.Float32).alias("2J"))

    energy_levels_zerodindexed = [
        FloersEnergyLevel(
            levelname=row["Configuration"],
            parity=row["Parity"],
            g=row["g"],
            energyabovegsinpercm=float(row["Energy"]),
        )
        for row in dflevels.iter_rows(named=True)
    ]

    energy_levels = [None, *energy_levels_zerodindexed]

    artisatomic.log_and_print(flog, f"Read {len(energy_levels[1:]):d} levels")

    transitions = []
    dftransitions = pl.from_pandas(pd.read_csv(lines_file, sep=r"\s+", skiprows=28, dtype_backend="pyarrow"))

    transitions = [
        FloersTransition(lowerlevel=lowerindex + 1, upperlevel=upperindex + 1, A=A, coll_str=-1)
        for lowerindex, upperindex, A in dftransitions[["Lower", "Upper", "A"]].iter_rows()
    ]

    transition_count_of_level_name = {
        config: (
            dftransitions.filter(pl.col("Config_Lower") == config).height
            + dftransitions.filter(pl.col("Config_Upper") == config).height
        )
        for config in dflevels["Configuration"]
    }

    # this check is slow
    # assert sum(transition_count_of_level_name.values()) == len(transitions) * 2

    artisatomic.log_and_print(flog, f"Read {len(transitions)} transitions")

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    part = levelname.split(".")[-1]
    if part[-1] not in "spdfg":
        # end of string is a number of electrons in the orbital, not a principal quantum number, so remove it
        assert part[-1].isdigit()
        part = part.rstrip("0123456789")
    return int(part.rstrip("spdfg"))