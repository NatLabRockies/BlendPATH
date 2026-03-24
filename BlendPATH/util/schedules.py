import csv
from collections import defaultdict

from importlib_resources import files

schedule_file = files("BlendPATH.util").joinpath("pipe_dimensions_metric.csv")

SCHEDULES_STEEL_DN = {}
SCHEDULES_STEEL_SCH = defaultdict(lambda: defaultdict(dict))
SCHEDULES_STEEL_ID = defaultdict(lambda: defaultdict(list))
SCHEDULES_PE_DN = {}
SCHEDULES_PE_ID = defaultdict(lambda: defaultdict(list))


with open(schedule_file, newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        SCHEDULES_STEEL_DN[float(row["Outer diameter [mm]"])] = float(row["DN"])

        for sch in row:
            if (sch not in ["DN", "Outer diameter [mm]"]) and (not row[sch] == ""):
                SCHEDULES_STEEL_SCH[float(row["DN"])][sch] = float(row[sch])
                SCHEDULES_STEEL_ID[float(row["DN"])]["thickness_mm"].append(
                    float(row[sch])
                )
                SCHEDULES_STEEL_ID[float(row["DN"])]["diameter_mm"].append(
                    float(row["Outer diameter [mm]"]) - 2 * float(row[sch])
                )
