from dataclasses import dataclass

import numpy as np


@dataclass
class pipe_dim:
    dn: float
    nps: float


# Based on https://www.octalsteel.com/steel-pipe-dimensions-sizes/ and https://en.wikipedia.org/wiki/Nominal_Pipe_Size
# OD (mm) is key, DN (metric) and NPS (in) is value
_OD_DN_NPS_MAP = {
    10.3: pipe_dim(dn=6.0, nps=0.125),
    13.7: pipe_dim(dn=8.0, nps=0.25),
    17.1: pipe_dim(dn=10.0, nps=0.375),
    21.3: pipe_dim(dn=15.0, nps=0.5),
    26.7: pipe_dim(dn=20.0, nps=0.75),
    33.4: pipe_dim(dn=25.0, nps=1.0),
    42.2: pipe_dim(dn=32.0, nps=1.25),
    48.3: pipe_dim(dn=40.0, nps=1.5),
    60.3: pipe_dim(dn=50.0, nps=2.0),
    73.0: pipe_dim(dn=65.0, nps=2.5),
    88.9: pipe_dim(dn=80.0, nps=3.0),
    101.6: pipe_dim(dn=90.0, nps=3.5),
    114.3: pipe_dim(dn=100.0, nps=4.0),
    125.0: pipe_dim(dn=110.0, nps=4.4),  # Added for selected cases
    141.3: pipe_dim(dn=125.0, nps=5.0),
    168.3: pipe_dim(dn=150.0, nps=6.0),
    180.0: pipe_dim(dn=180.0, nps=7.0),  # Added for selected cases
    219.1: pipe_dim(dn=200.0, nps=8.0),
    273.1: pipe_dim(dn=250.0, nps=10.0),
    323.9: pipe_dim(dn=300.0, nps=12.0),
    355.6: pipe_dim(dn=350.0, nps=14.0),
    406.4: pipe_dim(dn=400.0, nps=16.0),
    457.2: pipe_dim(dn=450.0, nps=18.0),
    508.0: pipe_dim(dn=500.0, nps=20.0),
    558.8: pipe_dim(dn=550.0, nps=22.0),
    609.6: pipe_dim(dn=600.0, nps=24.0),
    660.4: pipe_dim(dn=650.0, nps=26.0),
    711.2: pipe_dim(dn=700.0, nps=28.0),
    762.0: pipe_dim(dn=750.0, nps=30.0),
    812.8: pipe_dim(dn=800.0, nps=32.0),
    863.6: pipe_dim(dn=850.0, nps=34.0),
    914.4: pipe_dim(dn=900.0, nps=36.0),
    965.0: pipe_dim(dn=950.0, nps=38.0),
    1016.0: pipe_dim(dn=1000.0, nps=40.0),
    1066.8: pipe_dim(dn=1050.0, nps=42.0),
    1117.6: pipe_dim(dn=1100.0, nps=44.0),
    1168.4: pipe_dim(dn=1150.0, nps=46.0),
    1219.2: pipe_dim(dn=1200.0, nps=48.0),
    1320.8: pipe_dim(dn=1300.0, nps=52.0),
    1422.4: pipe_dim(dn=1400.0, nps=56.0),
    1524.0: pipe_dim(dn=1500.0, nps=60.0),
    1625.6: pipe_dim(dn=1600.0, nps=64.0),
    1727.2: pipe_dim(dn=1700.0, nps=68.0),
    1828.8: pipe_dim(dn=1800.0, nps=72.0),
}

_OD_LIST = np.asarray(list(_OD_DN_NPS_MAP.keys()))

_NPS_DN_MAP = {val.nps: val.dn for val in _OD_DN_NPS_MAP.values()}
_DN_NPS_MAP = {val.dn: val.nps for val in _OD_DN_NPS_MAP.values()}


def get_DN_NPS(od_mm: float) -> tuple[float, float]:
    i = _OD_LIST[np.abs(_OD_LIST - od_mm).argmin()]

    return _OD_DN_NPS_MAP[i].dn, _OD_DN_NPS_MAP[i].nps
