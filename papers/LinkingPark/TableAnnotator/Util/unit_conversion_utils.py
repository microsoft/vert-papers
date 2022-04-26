# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


class UnitConversionUtils:

    @staticmethod
    def convert_length_units(value):
        # Units in meters
        conversions = []
        for i in [1, 2, 3, 4, 6, 9, 12, 15, 21, 24]:
            conversions.append(value / 10 ** i)
            conversions.append(value * 10 ** i)
        # for i in [1, 2, 3, 6, 9, 12, 15, 21, 24]:
        #     conversions.append(pvalue*10**i)
        # Convert to meter, parsec, light-year, light-second, astronomical unit, miles, foot, inch, thou
        for val in [value, value / 3.086e16, value / 9.461e15, value / 2.998e8, value / 1.496e11,
                    value / 1609.34, value * 3.281, value * 39.37, value * 39270]:
            conversions.append(val)

        return conversions

    @staticmethod
    def convert_area_units(value):
        # Units in square metre
        conversions = [value]
        for i in [4, 6]:
            conversions.append(value * 10 ** i)
            conversions.append(value / 10 ** i)
        return conversions

    @staticmethod
    def convert_volume_units(value):
        # Units in cubic metre
        conversions = [value, value / 1e9]
        for i in [1, 2, 3, 5, 6, 9, 15, 18]:
            conversions.append(value * 10 ** i)
        return conversions

    @staticmethod
    def convert_frequency_units(value):
        # Units in hertz
        conversions = [value]
        for i in [3, 6, 9]:
            conversions.append(value / 10 ** i)

        return conversions

    @staticmethod
    def convert_time_units(value):
        # Units in seconds
        # Convert to seconds, minutes, hour, day, week, month, year
        conversions = [value, value / 60, value / 360, value / 86400, value / 604800, value / 2.628e6,
                       value / 3.154e7]
        for i in range(3, 19, 3):
            conversions.append(value * 10 ** i)
        return conversions

    @staticmethod
    def convert_mass_units(value):
        # Units in Kilogram
        # Convert to all other units
        conversions = [value, value * 2.205, value / 1.66e-27, value / 10]
        for i in range(3, 22, 3):
            conversions.append(value / 10 ** i)
        for i in range(1, 7):
            conversions.append(value * 10 ** i)
        for i in range(9, 28, 3):
            conversions.append(value * 10 ** i)
        return conversions

    @staticmethod
    # TODO: Im unsure whether this will work. The units in Wikidata seem quite obscure so we should check in the future
    def convert_concentration_units(value):
        return [value, value * 1e3, value * 1e6, value, value / 35.5, value, value, value, value, value, value, value]

    @staticmethod
    def check_mention_in_conversions(conversions, mention):
        for value in conversions:
            if np.isclose(float(value), float(mention), rtol=1e-2, equal_nan=False):
                return True
        return False

    @staticmethod
    def convert_temperature_units(value):
        # Convert to Celsius, Kelvin, Fahrenheit, Rankine
        conversions = [value, value + 273.15, (value * 9 / 5) + 32, (value * 9 / 5) + 491.67]
        return conversions
        # return UnitConversionUtils.check_mention_in_conversions(conversions, cell_mention)

    @staticmethod
    def convert_pressure_units(value):
        # Unit in standard atmosphere
        # Convert to standard atmosphere, technical atmosphere, bar, pascal, torr and millimeter of mercury, meter of water
        conversions = [value, value * 1.033, value * 1.013, value * 101325, value * 9.869, value / 101,
                       value / 1013, value * 760, value / 0.09869]
        return conversions

    @staticmethod
    def convert_speed_units(value):
        # kilometre per hour
        # Convert to to kilometre per hour, metre per second and Knot
        conversions = [value, value / 3600, value / 3.6, value / 1.852]
        return conversions
        # return UnitConversionUtils.check_mention_in_conversions(conversions, cell_mention)

    @staticmethod
    def convert_enthalpy_units(value):
        # Convert to joule per mole
        return [value, value * 1e3]
