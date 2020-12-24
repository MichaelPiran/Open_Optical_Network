# all mathematical conversion
import numpy as np


def conv_tera_to_unit(value_tera):
    return value_tera*1e12


def conv_giga_to_unit(value_giga):
    return value_giga*1e9


def conv_mega_to_unit(value_mega):
    return value_mega*1e6


def conv_kilo_to_unit(value_kilo):
    return value_kilo*1e3


def conv_milli_to_unit(value_milli):
    return value_milli*1e-3


def conv_micro_to_unit(value_micro):
    return value_micro*1e-6


def conv_nano_to_unit(value_nano):
    return value_nano*1e-9


def conv_pico_to_unit(value_pico):
    return value_pico*1e-12


def conv_db_to_linear(value_db):
    return 10**(value_db/10)


def conv_db_to_linear_power(value_db):
    return 10**(value_db/20)


def conv_linear_to_db(value_linear):
    return 10*np.log10(value_linear)


def conv_linear_to_db_power(value_linear):
    return 20*np.log10(value_linear)


def conv_dbm_to_linear(value_dbm):
    return 1e-3*10**(value_dbm/10)


def conv_dbm_to_linear_power(value_dbm):
    return 1e-3*10**(value_dbm/20)


def conv_linear_to_dbm(value_linear):
    return 10*np.log10(value_linear/1e-3)


def conv_linear_to_dbm_power(value_linear):
    return 20*np.log10(value_linear/1e-3)
