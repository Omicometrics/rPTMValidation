#! /usr/bin/env python3

import collections

Mass = collections.namedtuple('Mass', ['mono', 'avg'])

AA_MASSES = {
    'G': Mass(57.02146, 57.052),
    'A': Mass(71.03711, 71.078),
    'S': Mass(87.03203, 87.078),
    'P': Mass(97.05276, 97.117),
    'V': Mass(99.06841, 99.133),
    'T': Mass(101.04768, 101.105),
    'C': Mass(103.00918, 103.144),
    'I': Mass(113.08406, 113.160),
    'L': Mass(113.08406, 113.160),
    'N': Mass(114.04292, 114.104),
    'D': Mass(115.02693, 115.089),
    'Q': Mass(128.05857, 128.131),
    'K': Mass(128.09495, 128.174),
    'E': Mass(129.04258, 129.116),
    'M': Mass(131.04048, 131.198),
    'H': Mass(137.05891, 137.142),
    'F': Mass(147.06841, 147.177),
    'R': Mass(156.10110, 156.188),
    'Y': Mass(163.06332, 163.170),
    'W': Mass(186.07931, 186.213)
}

AA_SYMBOLS = {
    'Asparagine': 'N',
    'Alanine': 'A',
    'Arginine': 'R',
    'Aspartic acid': 'D',
    'Aspartate': 'D',
    'Cysteine': 'C',
    'Glutamic acid': 'E',
    'Glutamate': 'E',
    'Glutamine': 'Q',
    'Glycine': 'G',
    'Histidine': 'H',
    'Isoleucine': 'I',
    'Leucine': 'L',
    'Lysine': 'K',
    'Methionine': 'M',
    'Phenylalanine': 'F',
    'Proline': 'P',
    'Serine': 'S',
    'Threonine': 'T',
    'Tryptophan': 'W',
    'Tyrosine': 'Y',
    'Valine': 'V'
}

FIXED_MASSES = {
    "tag": 304.20536,
    "h20": 18.006067,
    "h": 1.0073,
    "cys": 57.021464
}

ELEMENT_MASSES = {
    'H': Mass(1.0073, 1.0079),
    '2H': Mass(2.0141, 2.0141),
    'Li': Mass(6.941, 2.0141),
    'C': Mass(12, 12.0107),
    "13C": Mass(13.0033, 13.0034),
    "N": Mass(14.0031, 14.0067),
    "15N": Mass(15.0001, 15.0001),
    "O": Mass(15.9949, 15.9994),
    "18O": Mass(17.99916, 17.99916),
    "F": Mass(18.99840, 18.99840),
    "Na": Mass(22.98977, 22.98977),
    "P": Mass(30.97376, 30.97376),
    "S": Mass(31.97207, 32.065),
    "Cl": Mass(34.96885, 35.453),
    "K": Mass(38.96371, 39.0983),
    "Ca": Mass(39.96259, 40.078),
    "Fe": Mass(55.93494, 55.845),
    "Ni": Mass(57.93535, 58.6934),
    "Zn": Mass(63.92914, 65.409),
    "Se": Mass(79.91652, 78.96),
    "Br": Mass(78.91834, 79.904),
    "Ag": Mass(106.90509, 107.8682),
    "Hg": Mass(201.97062, 200.59),
    "Au": Mass(196.96654, 196.96655),
    "I": Mass(126.90447, 126.90447),
    "Mo": Mass(97.9054073, 95.94),
    "Cu": Mass(62.9295989, 63.546),
    "e": Mass(0.000549, 0.000549),
    "B": Mass(11.0093055, 10.811),
    "As": Mass(74.9215942, 74.9215942),
    "Cd": Mass(113.903357, 112.411),
    "Cr": Mass(51.9405098, 51.9961),
    "Co": Mass(58.9331976, 58.933195),
    "Mn": Mass(54.9380471, 54.938045),
    "Mg": Mass(23.9850423, 24.305),
    "Pd": Mass(105.903478, 106.42),
    "Al": Mass(26.9815386, 26.9815386)
}