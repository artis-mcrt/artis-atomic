import os.path
import typing as t
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from astropy import constants as const
from astropy import units as u

import artisatomic

hc_in_ev_cm = (const.h * const.c).to("eV cm").value
hc_in_ev_angstrom = (const.h * const.c).to("eV angstrom").value


class EnergyLevel(t.NamedTuple):
    levelname: str
    energyabovegsinpercm: float
    g: float
    parity: float


class TransitionTuple(t.NamedTuple):
    lowerlevel: int
    upperlevel: int
    A: float
    coll_str: float


datafilepath = Path(os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-mons")

# outggf_Ln_V-VII.zip folder:
#     45 files outggf for each lanthanide between the V and VII spectra:
#     first column is wavelength of the E1 transition (A),
#     second column is the lower energy level of the transition (1000 cm^-1)
#     third column is the oscillator strength
#
# outglv_Ln_V--VII.zip folder:
#     45 files outglv for each lanthanide between the V and VII spectra:
#     first column is the energy of levels (1000 cm^-1)
#     second column is the total angular momentum (J-value)


def extend_ion_list(ion_handlers):
    # Data files contain La-Lu V-VII ions
    Z_indatafile = range(57, 72)
    ions_indatafile = [5, 6, 7]

    for Z in Z_indatafile:
        for ion in ions_indatafile:
            atomic_number = Z
            ion_stage = ion
            found_element = False
            for tmp_atomic_number, list_ions_handlers in ion_handlers:
                if tmp_atomic_number == atomic_number:
                    # add an ion that is not present in the element's list
                    if ion_stage not in [x[0] if hasattr(x, "__getitem__") else x for x in list_ions_handlers]:
                        list_ions_handlers.append((ion_stage, "mons"))
                        list_ions_handlers.sort(key=lambda x: x[0] if hasattr(x, "__getitem__") else x)
                    found_element = True

            if not found_element:
                ion_handlers.append(
                    (
                        atomic_number,
                        [(ion_stage, "mons")],
                    )
                )
    ion_handlers.sort(key=lambda x: x[0])
    return ion_handlers


def get_transition_data(atomic_number, ion_stage, energiesabovegsinpercm, g_arr, parquet_filepath, flog):
    ziparchive_outggf = zipfile.ZipFile(datafilepath / "outggf_Ln_V--VII.zip", "r")
    datafilename_transitions = (
        f"outggf_Ln_V--VII/outggf_sorted_{artisatomic.elsymbols[atomic_number]}_{artisatomic.roman_numerals[ion_stage]}"
    )

    with ziparchive_outggf.open(datafilename_transitions) as datafile_transitions:
        transition_wavelength_A, energy_levels_lower_1000percm, oscillator_strength = np.loadtxt(
            datafile_transitions, unpack=True, delimiter=","
        )
    artisatomic.log_and_print(flog, f"transitions: {len(energy_levels_lower_1000percm)}")

    energy_levels_lower_percm = energy_levels_lower_1000percm * 1000

    # Get index of lower level of transition
    lowerlevels = [
        (
            np.abs(energiesabovegsinpercm - energylevellower)  # get closest energy in energy level array to lower level
        ).argmin()  # then get the index with argmin()
        for energylevellower in energy_levels_lower_percm
    ]

    # get energy of upper level of transition
    energy_levels_lower_ev = energy_levels_lower_percm * hc_in_ev_cm
    transitionenergyev = hc_in_ev_angstrom / transition_wavelength_A

    energy_levels_upper_ev = transitionenergyev + energy_levels_lower_ev
    energy_levels_upper_percm = energy_levels_upper_ev / hc_in_ev_cm

    # Get index of upper level of transition
    upperlevels = [
        (
            np.abs(energiesabovegsinpercm - energylevelupper)  # get closest energy in energy level array
        ).argmin()  # then get the index with argmin()
        for energylevelupper in energy_levels_upper_percm
    ]

    # Get A value from oscillator strength
    OSCSTRENGTHCONVERSION = 1.3473837e21
    c_angps = 2.99792458e18
    A_ul = np.array(
        [
            osc / (g_arr[upper] / g_arr[lower] * OSCSTRENGTHCONVERSION / (c_angps / lambda_A) ** 2)
            for lambda_A, osc, lower, upper in zip(
                transition_wavelength_A, oscillator_strength, lowerlevels, upperlevels, strict=False
            )
        ]
    )

    dict_transitions = {
        "lowerlevels": lowerlevels,
        "upperlevels": upperlevels,
        "oscillator_strength": oscillator_strength,
        "g_lower": [g_arr[lower] for lower in lowerlevels],
        "A_ul": A_ul,
        "energy_lower_level_ev": energy_levels_lower_ev,
        "transitionenergyev": transitionenergyev,
    }
    df_transitions = pd.DataFrame.from_dict(dict_transitions)

    n_transitions = len(df_transitions)
    # n_transitions = df_transitions.shape[0]
    assert n_transitions == len(
        energy_levels_lower_1000percm
    )  # check number of transitions is the same as the number read in

    # Save DataFrame to a Parquet file
    df_transitions.to_parquet(parquet_filepath, engine="pyarrow")  # or engine="fastparquet"
    print(f"Parquet file created ({parquet_filepath})")
    # quit()

    return df_transitions, n_transitions


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    # Read first file
    ziparchive_outglv = zipfile.ZipFile(datafilepath / "outglv_Ln_V--VII.zip", "r")
    datafilename_energylevels = (
        f"outglv_Ln_V--VII/outglv_0_{artisatomic.elsymbols[atomic_number]}_{artisatomic.roman_numerals[ion_stage]}"
    )

    with ziparchive_outglv.open(datafilename_energylevels) as datafile_energylevels:
        energy_levels1000percm, j_arr = np.loadtxt(datafile_energylevels, unpack=True, delimiter=",")
    artisatomic.log_and_print(flog, f"levels: {len(energy_levels1000percm)}")

    # Sort table by energy levels
    dfenergylevels = pd.DataFrame.from_dict(
        {"energiesabovegsinpercm": energy_levels1000percm * 1000, "g": 2 * j_arr + 1}
    ).sort_values("energiesabovegsinpercm")

    energiesabovegsinpercm = dfenergylevels["energiesabovegsinpercm"].to_numpy()
    g_arr = dfenergylevels["g"].to_numpy()

    parity = None  # Only E1 so always allowed transitions.

    energy_levels = [None] + [
        EnergyLevel(
            levelname=str(energyabovegsinpercm),
            parity=parity,
            g=g,
            energyabovegsinpercm=energyabovegsinpercm,
        )
        for g, energyabovegsinpercm in zip(g_arr, energiesabovegsinpercm, strict=True)
    ]

    # Get next file
    parquet_filename = f"outggf_{atomic_number}_{ion_stage}.parquet"
    parquet_filepath = datafilepath / parquet_filename
    if parquet_filepath.is_file():
        df_transitions = pd.read_parquet(parquet_filepath, engine="pyarrow")
        n_transitions = len(df_transitions)
        artisatomic.log_and_print(
            flog, f"Read from {parquet_filename} \n"
                  f"transitions: {n_transitions}"
        )
    else:
        df_transitions, n_transitions = get_transition_data(atomic_number, ion_stage,
                                                            energiesabovegsinpercm, g_arr, parquet_filepath, flog)

    # Get ionization energy
    ionization_energy_in_ev_nist = artisatomic.get_nist_ionization_energies_ev()[(atomic_number, ion_stage)]

    ionization_energy_in_ev = max(df_transitions["transitionenergyev"])
    artisatomic.log_and_print(
        flog, f"ionization energy: {ionization_energy_in_ev} eV (NIST: {ionization_energy_in_ev_nist} eV)"
    )

    # If ionisation potential in data does not match NIST to within 1 decimal place
    # then use NIST instead (probably more accurate?)
    if abs(ionization_energy_in_ev - ionization_energy_in_ev_nist) > 0.1:
        ionization_energy_in_ev = ionization_energy_in_ev_nist
        artisatomic.log_and_print(
            flog, f"Energies do not match -- using NIST value of {ionization_energy_in_ev_nist} eV"
        )
        if abs(ionization_energy_in_ev - ionization_energy_in_ev_nist) > 5:
            print("Energies really do not match -- check using correct parquet file")
            quit()

    df_transitions = pl.from_pandas(df_transitions)

    cut_on_log_gf = True
    if cut_on_log_gf:
        # df_transitions["log(gf)"] = np.log10(df_transitions["oscillator_strength"] * df_transitions["g_lower"])
        df_transitions = df_transitions.with_columns(
            (pl.col("oscillator_strength") * pl.col("g_lower")).log10().alias("log(gf)")
        )
        cut_value = -3
        # df_transitions = df_transitions[df_transitions["log(gf)"] >= cut_value]
        df_transitions = df_transitions.filter(pl.col("log(gf)") >= cut_value)
        # n_new_transitions = len(df_transitions)
        n_new_transitions = df_transitions.shape[0]
        artisatomic.log_and_print(
            flog,
            f"Cut placed to reduce number of transitions: log(gf) > {cut_value} \n"
            f"{n_transitions} transitions reduced to {n_new_transitions} transitions"
            f" (removed {n_transitions-n_new_transitions})",
        )

    cut_on_excitation_energy = True
    if cut_on_excitation_energy:
        cut_temperature = 50000  # K

        KB = 8.617e-5  # /// Boltzmann constant eV/K
        thermal_energy = KB * cut_temperature

        n_old_transitions = df_transitions.shape[0]
        # df_transitions = df_transitions[df_transitions["energy_lower_level_ev"] < thermal_energy]
        df_transitions = df_transitions.filter(pl.col("energy_lower_level_ev") < thermal_energy)
        n_new_transitions = df_transitions.shape[0]
        artisatomic.log_and_print(
            flog,
            f"Cut placed to reduce number of transitions: lower level energy < kT "
            f"(T={cut_temperature} K, kT={thermal_energy} eV) \n"
            f"{n_old_transitions} transitions reduced to {n_new_transitions} transitions"
            f" (removed {n_old_transitions - n_new_transitions})",
        )

    # transitions = [
    #     TransitionTuple(
    #         lowerlevel=int(lower) + 1,
    #         upperlevel=int(upper) + 1,
    #         A=A,
    #         coll_str=-1,
    #     )
    #     for A, lower, upper in df_transitions[["A_ul", "lowerlevels", "upperlevels"]].to_numpy()
    # ]

    transitions = [
        TransitionTuple(
            lowerlevel=int(row["lowerlevels"]) + 1,
            upperlevel=int(row["upperlevels"]) + 1,
            A=row["A_ul"],
            coll_str=-1,
        )
        for row in df_transitions.to_dicts()
    ]

    transition_count_of_level_name = defaultdict(int)
    # for lower, upper in zip(lowerlevels, upperlevels, strict=True):
    #     transition_count_of_level_name[energy_levels[lower + 1].levelname] += 1
    #     transition_count_of_level_name[energy_levels[upper + 1].levelname] += 1
    for row in df_transitions.iter_rows(named=True):
        lower = (row["lowerlevels"])
        upper = (row["upperlevels"])
        transition_count_of_level_name[energy_levels[lower + 1].levelname] += 1
        transition_count_of_level_name[energy_levels[upper + 1].levelname] += 1

    if cut_on_log_gf:
        assert len(transitions) == n_new_transitions
    else:
        assert len(transitions) == n_transitions
    # check number of transitions is what we expect

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name


# read_levels_and_transitions(atomic_number=57, ion_stage=5, flog=None)
