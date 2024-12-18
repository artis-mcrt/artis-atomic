import typing as t
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars as pl
from xopen import xopen

import artisatomic

# the h5 file comes from Andreas Floers's DREAM parser
jpltpath = (Path(__file__).parent.resolve() / ".." / "atomic-data-tanaka-jplt" / "data_v1.1").resolve()
hc_in_ev_cm = 0.0001239841984332003


class EnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


def extend_ion_list(ion_handlers):
    tanakaions = sorted(
        [tuple(int(x) for x in f.parts[-1].split(".")[0].split("_")) for f in jpltpath.glob("*_*.txt*")]
    )

    for atomic_number, ion_stage in tanakaions:
        found_element = False
        for tmp_atomic_number, list_ions_handlers in ion_handlers:
            if tmp_atomic_number == atomic_number:
                # add an ion that is not present in the element's list
                if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                    list_ions_handlers.append((ion_stage, "tanakajplt"))
                    list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                found_element = True

        if not found_element:
            ion_handlers.append(
                (
                    atomic_number,
                    [(ion_stage, "tanakajplt")],
                )
            )

    ion_handlers.sort(key=lambda x: x[0])

    return ion_handlers


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    filename = f"{atomic_number}_{ion_stage}.txt"
    if Path(jpltpath / f"{filename}.zst").is_file():
        filename = f"{filename}.zst"
    print(f"Reading Tanaka et al. Japan-Lithuania database for Z={atomic_number} ion_stage {ion_stage} from {filename}")
    with xopen(jpltpath / filename) as fin:
        artisatomic.log_and_print(flog, fin.readline().strip())
        artisatomic.log_and_print(flog, fin.readline().strip())
        artisatomic.log_and_print(flog, fin.readline().strip())
        assert fin.readline().strip() == f"# {atomic_number} {ion_stage}"
        levelcount, transitioncount = (int(x) for x in fin.readline().removeprefix("# ").split())
        artisatomic.log_and_print(flog, f"levels: {levelcount}")
        artisatomic.log_and_print(flog, f"transitions: {transitioncount}")

        fin.readline()
        str_ip_line = fin.readline()
        ionization_energy_in_ev = float(str_ip_line.removeprefix("# IP = "))
        artisatomic.log_and_print(flog, f"ionization energy: {ionization_energy_in_ev} eV")
        assert fin.readline().strip() == "# Energy levels"
        assert fin.readline().strip() == "# num  weight parity      E(eV)      configuration"

        with pd.read_fwf(
            fin,
            chunksize=levelcount,
            nrows=levelcount,
            colspecs=[(0, 7), (7, 15), (15, 19), (19, 34), (34, None)],
            names=["num", "g", "parity", "energy_ev", "configuration"],
        ) as reader:
            # dflevels = pd.concat(reader, ignore_index=True)

            dflevels = artisatomic.add_dummy_zero_level(
                pl.from_pandas(reader.get_chunk(levelcount)).with_columns(pl.col("g").cast(pl.Float64))
            )
            # print(dflevels)

            energy_levels = [None]
            for row in dflevels[1:].iter_rows(named=True):
                parity = 1 if row["parity"].strip() == "odd" else 0
                energyabovegsinpercm = float(row["energy_ev"] / hc_in_ev_cm)

                levelname = f"{row['num']},{row['parity']},{row['configuration'].strip()}"
                energy_levels.append(
                    EnergyLevel(
                        levelname=levelname, parity=parity, g=row["g"], energyabovegsinpercm=energyabovegsinpercm
                    )
                )

        assert (dflevels.height - 1) == levelcount

        line = fin.readline().strip()
        assert line in ("# Transitions", "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)")
        if line == "# Transitions":
            assert fin.readline().strip() == "# num_u   num_l   wavelength(nm)     g_u*A      log(g_l*f)"
        dftransitions = pl.from_pandas(
            pd.read_fwf(
                fin,
                colspecs=[(0, 7), (7, 15), (15, 30), (30, 43), (43, None)],
                names=["upperlevel", "lowerlevel", "wavelength", "g_u_times_A", "log(g_l*f)"],
                dtype_backend="pyarrow",
            )
        ).select(
            pl.col("lowerlevel").cast(pl.Int64),
            pl.col("upperlevel").cast(pl.Int64),
            pl.col("g_u_times_A").cast(pl.Float64),
        )

    transition_count_of_level_name = defaultdict(int)

    for upperlevel, lowerlevel in dftransitions[["upperlevel", "lowerlevel"]].iter_rows(named=False):
        transition_count_of_level_name[energy_levels[upperlevel].levelname] += 1
        transition_count_of_level_name[energy_levels[lowerlevel].levelname] += 1

    assert dftransitions.height == transitioncount
    dftransitions = dftransitions.with_columns(
        g_u=pl.col("upperlevel").map_elements(lambda upperlevel: energy_levels[upperlevel].g, return_dtype=pl.Float64)
    ).with_columns(A=pl.col("g_u_times_A") / pl.col("g_u"))
    dftransitions = dftransitions.select(["lowerlevel", "upperlevel", "A"])

    return ionization_energy_in_ev, energy_levels, dftransitions, transition_count_of_level_name


def get_level_valence_n(levelname: str):
    n = int(levelname.split("  ")[-1].split(" ", maxsplit=1)[0].rstrip("spdfg+-"))
    assert n >= 0
    assert n < 20
    return n
