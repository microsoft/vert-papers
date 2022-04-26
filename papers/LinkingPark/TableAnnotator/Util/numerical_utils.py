# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import datetime
import json
import logging

import numpy as np
from dateutil.relativedelta import *

from TableAnnotator.Config import config
from TableAnnotator.Util.unit_conversion_utils import UnitConversionUtils
from TableAnnotator.Config import shortname

logger = logging.getLogger(__name__)


class NumericalPropertyLinkingUtils:
    def __init__(self, units_dir, factors_dir):
        with open(units_dir, 'r') as fp:
            self.qnumber_to_unit = json.load(fp)
        with open(factors_dir, 'r') as fp:
            self.unit_to_standard_value_factor = json.load(fp)

    @staticmethod
    def fuzzy_match_dates(cell_value, property_value, threshold):
        try:
            cell_date = datetime.date.fromisoformat(cell_value.split("T")[0][1:])
            property_date = datetime.date.fromisoformat(property_value.split("T")[0][1:])
        except ValueError:
            return False
        try:
            range_start = property_date + relativedelta(days=-threshold)
        except ValueError:
            range_start = property_date
        except OverflowError:
            range_start = property_date
        try:
            range_end = property_date + relativedelta(days=+threshold)
        except OverflowError:
            range_end = property_date
        if range_start <= cell_date <= range_end:
            return True
        else:
            return False

    def get_conversions(self, pnumber, pvalue, punits):
        property_types = {
            "convert_length_units": {"P2120", "P2073", "P2050", "P2048", "P2049", "P2043", "P2044", "P2148", "P2151"},
            # All area(P2046, P2053, P2112), Volume (P2234), and Frequency (P2114) conversions are in multiples of ten
            "convert_area_units": {"P2046", "P2053", "P2112"},
            "convert_volume_units": {"P2234"},
            "convert_frequency_units": {"P2114"},
            # All volume unit conversions are in multiples of ten
            "convert_time_units": {"P2047", "P2114"},
            "convert_mass_units": {"P2067"},
            "convert_concentration_units": {"P2177", "P2202", "P2203", "P2240", "P2300", "P2404", "P2405", "P2406",
                                            "P2407"},
            "convert_temperature_units": {"P2076", "P2101", "P2102", "P2107", "P2113", "P2128"},
            "convert_pressure_units": {"P2077", "P2119"},
            "convert_speed_units": {"P2052", "P2075"},
            "convert_enthalpy_units": {"P2066", "P2116", "P2117"},
        }

        try:
            units = self.qnumber_to_unit[punits]
            standard_value = pvalue * self.unit_to_standard_value_factor[units]
        except KeyError:
            return [pvalue]
        # cannot use factors for following units
        if units == 'kelvin':
            standard_value -= 273.15
        elif units == 'degree Fahrenheit':
            standard_value -= 32
        elif units == 'Rankine scale':
            standard_value -= 491.67

        for property_type, properties in property_types.items():
            if pnumber in properties:
                method_to_call = getattr(UnitConversionUtils, property_type)
                conversions = method_to_call(standard_value)
                return conversions

        return [pvalue]

    def match_numbers_with_conversion(self, pnumber, pvalue, punits, cell_mention):
        conversions = self.get_conversions(pnumber, pvalue, punits)
        if not conversions:
            return False
        return UnitConversionUtils.check_mention_in_conversions(conversions, cell_mention)

    def check_cell_mention_in_range(self, pnumber, range_min, range_max, punits, cell_mention):
        min_values = self.get_conversions(pnumber, range_min, punits)
        max_values = self.get_conversions(pnumber, range_max, punits)

        for min_val, max_val in zip(min_values, max_values):
            if min_val <= float(cell_mention) <= max_val:
                return True
        return False



def is_date(string):
    if string == "":
        return False
    if string[0] == '-' or string[0] == '+':
        string = string[1:]
    if '/' in string:
        string = string.replace('/', '-')
    try:
        datetime.datetime.strptime(string, '%Y-%m-%dT%H:%M:%SZ')
        return True
    except ValueError:
        try:
            datetime.datetime.strptime(string, "%Y-%m-%d")
            return True
        except ValueError:
            return False


def clear_zero_valued_times(string):
    times = string.split("T")
    date = times[0].split("-")
    for i in range(len(date)):
        if date[i] == "00":
            date[i] = "01"

    return "-".join(d for d in date) + "T" + times[1]


def fuzzy_match_dates(cell_value, property_value, threshold):
    cell_date = datetime.date.fromisoformat(cell_value.split("T")[0])
    try:
        property_date = datetime.date.fromisoformat(property_value.split("T")[0])
    except ValueError:
        # potential issue with feb 29
        date_list = property_value.split("T")[0].split('-')
        day = date_list[2]
        month = date_list[1]
        year = date_list[0]
        if month == '02' and day == '29':
            property_date = datetime.date.fromisoformat(year + '-' + month + '-28')
        else:
            return False
    try:
        range_start = property_date + relativedelta(days=-threshold)
    except ValueError:
        range_start = property_date
    except OverflowError:
        range_start = property_date
    try:
        range_end = property_date + relativedelta(days=+threshold)
    except OverflowError:
        range_end = property_date
    if range_start <= cell_date <= range_end:
        return True
    else:
        return False


class NumericalPropertyLinker:
    def __init__(self, use_characteristics):
        if use_characteristics:
            with open(config.property_entity_types_fn, 'r', encoding='utf-8') as fp:
                self.entity_types = json.load(fp)

            with open(config.type_property_stats_fn, 'r') as fp:
                self.type_property_stats = json.load(fp)

        self.numerical_property_linker = NumericalPropertyLinkingUtils(
            config.qnumbers_to_units_fn,
            config.unit_to_standard_unit_factors_fn)

    def is_match(self, candidate, property_number, property_type, property_value, property_units, cell_mention,
                 match_type, use_characteristics=False):
        if property_type == shortname.STRING or property_type == shortname.MONO:
            if match_type == "Direct Match" and property_value == cell_mention:
                return True
        elif property_type == shortname.QUANTITY:
            try:
                if match_type == "Direct Match" and np.isclose(float(property_value), float(cell_mention),
                                                               rtol=1e-2, equal_nan=False):
                    return True
                # Try to match with unit conversion
                elif match_type == "Fuzzy Match" and \
                        self.numerical_property_linker.match_numbers_with_conversion(property_number,
                                                                                     float(property_value),
                                                                                     property_units,
                                                                                     float(cell_mention)):
                    return True
                # Try to match numbers within a range
                elif match_type == "Fuzzy Match" and use_characteristics:
                    characteristics = self.type_property_stats[self.entity_types[candidate]][property_number]
                    if self.numerical_property_linker.check_cell_mention_in_range(property_number,
                                                                                  characteristics['min'],
                                                                                  characteristics['max'],
                                                                                  characteristics['unit'],
                                                                                  cell_mention):
                        return True
            except:
                pass
        elif property_type == shortname.TIME:
            if is_date(cell_mention):
                if cell_mention[0] == '-' or cell_mention[0] == '+':
                    cell_mention = cell_mention[1:]
                if '/' in cell_mention:
                    cell_mention = cell_mention.replace('/', '-')
                cell_mention = cell_mention.split('T')[0]
                date = datetime.datetime.strptime(
                    cell_mention, "%Y-%m-%d"
                )
                iso_date = date.isoformat() + "Z"
                property_value = clear_zero_valued_times(property_value[1:])
                if match_type == "Direct Match" and property_value == iso_date:
                    return True
                elif match_type == "Fuzzy Match" and fuzzy_match_dates(iso_date, property_value, 10):
                    return True
        return False