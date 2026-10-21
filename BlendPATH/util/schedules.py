import csv
from collections import defaultdict

import numpy as np
from importlib_resources import files

schedule_file = files("BlendPATH.util").joinpath("pipe_dimensions_metric.csv")
schedule_pe_file = files("BlendPATH.util").joinpath("pe_pipe_table.csv")

SCHEDULES_STEEL_DN = {}
SCHEDULES_STEEL_SCH = defaultdict(lambda: defaultdict(dict))
SCHEDULES_STEEL_ID = defaultdict(lambda: defaultdict(list))
SCHEDULES_PE_DN = {}
SCHEDULES_PE_ID = defaultdict(lambda: defaultdict(list))
POLYMER_SDR_TABLE = {}


with open(schedule_file, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        SCHEDULES_STEEL_DN[float(row["Outer diameter [mm]"])] = float(row["DN"])

        for sch in row:
            if (sch not in ["DN", "Outer diameter [mm]"]) and (row[sch] != ""):
                SCHEDULES_STEEL_SCH[float(row["DN"])][sch] = float(row[sch])
                SCHEDULES_STEEL_ID[float(row["DN"])]["thickness_mm"].append(
                    float(row[sch])
                )
                SCHEDULES_STEEL_ID[float(row["DN"])]["diameter_mm"].append(
                    float(row["Outer diameter [mm]"]) - 2 * float(row[sch])
                )

with open(schedule_pe_file, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        SCHEDULES_PE_DN[float(row["od_mm"])] = float(row["nom_size_in"])
        SCHEDULES_PE_ID[float(row["nom_size_in"])]["diameter_mm"].append(
            float(row["diameter_mm"])
        )
        SCHEDULES_PE_ID[float(row["nom_size_in"])]["thickness_mm"].append(
            float(row["t_min_mm"])
        )
        POLYMER_SDR_TABLE[
            (round(float(row["od_mm"]), 2), round(float(row["t_min_mm"]), 2))
        ] = float(row["dr"]) if row["dr"] != "" else None

for i, v in SCHEDULES_PE_ID.items():
    combined_lists = zip(
        SCHEDULES_PE_ID[i]["diameter_mm"], SCHEDULES_PE_ID[i]["thickness_mm"]
    )
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0])
    diameters, ths = zip(*sorted_combined_lists)
    SCHEDULES_PE_ID[i]["diameter_mm"] = np.asarray(list(diameters))
    SCHEDULES_PE_ID[i]["thickness_mm"] = np.asarray(list(ths))
