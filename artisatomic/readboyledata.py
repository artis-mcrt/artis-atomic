import h5py
from collections import defaultdict, namedtuple
from astropy import constants as const
import artisatomic
# from astropy import units as u
import os.path


filename_aoife_dataset = h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atomic-data-helium-boyle/aoife.hdf5"), 'r')

hc_in_ev_cm = (const.h * const.c).to('eV cm').value


def read_ionization_data(atomic_number, ion_stage):
    ionization_data = filename_aoife_dataset['/ionization_data'].value

    ionization_dict = {}
    for atomic_num, ion_number, ionization_energy in ionization_data:
        ion_dict = {ion_number: ionization_energy}
        if atomic_num in ionization_dict:
            ionization_dict[atomic_num].update(ion_dict)
        else:
            ionization_dict[atomic_num] = ion_dict
    ionization_dict[2][3] = 999999.  # He III

    return ionization_dict[atomic_number][ion_stage]


def read_levels_data(atomic_number, ion_stage):
    levels_data = filename_aoife_dataset['/levels_data'].value

    energy_levels = ['IGNORE']
    # TODO energyinRydbergs change back to energyabovegsinpercm
    energy_level_row = namedtuple("energylevel", 'atomic_number ion_number level_number energy g metastable energyabovegsinpercm parity levelname')

    for rowtuple in levels_data:
        atomic_num, ion_number, level_number, energy, g, metastable = rowtuple
        if energy == 0.0:
            energyabovegsinpercm = 0.0
        else:
            # energyabovegsinpercm = hc_in_ev_cm / energy
            energyabovegsinpercm = energy
        energy_level = energy_level_row(*rowtuple, energyabovegsinpercm, 0, f'level{level_number:05d}')

        if int(energy_level.atomic_number) != atomic_number or int(energy_level.ion_number) != ion_stage - 1:
            continue
        energy_levels.append(energy_level)

    return energy_levels


def read_lines_data(atomic_number, ion_stage):
    lines_data = (filename_aoife_dataset['/lines_data'].value)

    transitions = []
    transition_count_of_level_name = defaultdict(int)
    lines_row = namedtuple("transition", 'atomic_number ion_stage lowerlevel upperlevel A lambdaangstrom coll_str')

    for rowtuple in lines_data:
        (line_id, wavelength, atomic_num, ion_number, f_ul, f_lu,
         level_number_lower, level_number_upper, nu, B_lu, B_ul, A_ul) = rowtuple

        coll_str = -1  # TODO
        line = lines_row(atomic_num, ion_number, int(level_number_lower + 1), int(level_number_upper + 1),
                         A_ul, wavelength, coll_str)
        if int(atomic_num) != atomic_number or int(ion_number) != ion_stage - 1:
            continue
        # print(line)
        transition_count_of_level_name[level_number_lower] += 1
        transition_count_of_level_name[level_number_upper] += 1

        transitions.append(line)

    return transitions, transition_count_of_level_name


def read_levels_and_transitions(atomic_number, ion_stage, flog):
    assert atomic_number == 2
    # energy_levels = ['IGNORE']
    # artisatomic.log_and_print(flog, 'Reading atomic-data-He')
    transitions, transition_count_of_level_name = read_lines_data(atomic_number, ion_stage)

    ionization_energy_in_ev = read_ionization_data(atomic_number, ion_stage)
    # ionization_energy_in_ev = -1

    energy_levels = read_levels_data(atomic_number, ion_stage)
    # artisatomic.log_and_print(flog, f'Read {len(energy_levels[1:]):d} levels')

    return ionization_energy_in_ev, energy_levels, transitions, transition_count_of_level_name