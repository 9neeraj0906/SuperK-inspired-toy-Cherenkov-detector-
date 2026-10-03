from geant4_pybind import *
import csv

from detector import CherenkovDetectorConstruction
from Physics import PhysicsList
from Primary import PrimaryGeneratorAction
from Actions import EventAction, TrackingAction, SteppingAction


# ============================================================
# Run Manager
# ============================================================

runManager = G4RunManager()


# ============================================================
# Detector
# ============================================================

detector = CherenkovDetectorConstruction()

runManager.SetUserInitialization(
    detector
)


# ============================================================
# Physics
# ============================================================

runManager.SetUserInitialization(
    PhysicsList()
)


# ============================================================
# Primary Generator
# ============================================================

primaryGenerator = (
    PrimaryGeneratorAction()
)

runManager.SetUserAction(
    primaryGenerator
)


# ============================================================
# Event Action
# ============================================================

eventAction = EventAction(
    detector.pmtSD
)

runManager.SetUserAction(
    eventAction
)


# ============================================================
# Tracking Action
# ============================================================

trackingAction = TrackingAction(
    eventAction
)

runManager.SetUserAction(
    trackingAction
)


# ============================================================
# Stepping Action
# ============================================================

steppingAction = SteppingAction(
    eventAction
)

runManager.SetUserAction(
    steppingAction
)


# ============================================================
# Initialize
# ============================================================

runManager.Initialize()


# ============================================================
# Run
# ============================================================

runManager.BeamOn(1000)


# ============================================================
# EVENT TRUTH
# ============================================================

with open(
    "event_truth.csv",
    "w",
    newline=""
) as f:

    fieldnames = [

        "event",

        "initial_x_cm",
        "initial_y_cm",
        "initial_z_cm",

        "initial_dx",
        "initial_dy",
        "initial_dz",

        "water_entry_x_cm",
        "water_entry_y_cm",
        "water_entry_z_cm",

        "water_exit_x_cm",
        "water_exit_y_cm",
        "water_exit_z_cm",

        "track_length_cm"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in eventAction.event_truth:

        writer.writerow({

            "event":
                row["event"],

            "initial_x_cm":
                row["x"],

            "initial_y_cm":
                row["y"],

            "initial_z_cm":
                row["z"],

            "initial_dx":
                row["dx"],

            "initial_dy":
                row["dy"],

            "initial_dz":
                row["dz"],

            "water_entry_x_cm":
                row["water_in_x"],

            "water_entry_y_cm":
                row["water_in_y"],

            "water_entry_z_cm":
                row["water_in_z"],

            "water_exit_x_cm":
                row["water_out_x"],

            "water_exit_y_cm":
                row["water_out_y"],

            "water_exit_z_cm":
                row["water_out_z"],

            "track_length_cm":
                row["track_length"]
        })


# ============================================================
# PMT HITS
# ============================================================

with open(
    "pmt_hits.csv",
    "w",
    newline=""
) as f:

    fieldnames = [

        "event",
        "track_id",
        "parent_id",
        "pmt_id",

        "x_cm",
        "y_cm",
        "z_cm",

        "time_ns",
        "energy_eV"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in detector.pmtSD.all_hits:

        writer.writerow({

            "event":
                row["event"],

            "track_id":
                row["track_id"],

            "parent_id":
                row["parent_id"],

            "pmt_id":
                row["pmt_id"],

            "x_cm":
                row["x"],

            "y_cm":
                row["y"],

            "z_cm":
                row["z"],

            "time_ns":
                row["time"],

            "energy_eV":
                row["energy"]
        })


# ============================================================
# PHOTON TRUTH
# ============================================================

with open(
    "photon_truth.csv",
    "w",
    newline=""
) as f:

    fieldnames = [

        "event",
        "photon_id",
        "track_id",
        "parent_id",
        "creator",

        "creation_x_cm",
        "creation_y_cm",
        "creation_z_cm",

        "direction_x",
        "direction_y",
        "direction_z",

        "energy_eV",

        "muon_direction_x",
        "muon_direction_y",
        "muon_direction_z",

        "muon_x_cm",
        "muon_y_cm",
        "muon_z_cm",

        "hit_pmt_id",
        "hit_x_cm",
        "hit_y_cm",
        "hit_z_cm",
        "hit_time_ns"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()

    for row in eventAction.optical_truth:

        writer.writerow(row)


# ============================================================
# MATCH PMT HITS DIRECTLY TO PHOTON TRACKS
#
# This is NOT a reconstruction.
#
# Every optical photon already has its real Geant4 track ID.
# PMT SD records the same track ID.
# ============================================================

photon_by_track = {}

for row in eventAction.optical_truth:

    key = (
        row["event"],
        row["track_id"]
    )

    photon_by_track[key] = row


matched_hits = []


for hit in detector.pmtSD.all_hits:

    key = (
        hit["event"],
        hit["track_id"]
    )

    photon = photon_by_track.get(
        key
    )

    if photon is None:

        continue

    # --------------------------------------------------------
    # Attach PMT information directly to photon truth.
    # --------------------------------------------------------

    photon["hit_pmt_id"] = (
        hit["pmt_id"]
    )

    photon["hit_x_cm"] = (
        hit["x"]
    )

    photon["hit_y_cm"] = (
        hit["y"]
    )

    photon["hit_z_cm"] = (
        hit["z"]
    )

    photon["hit_time_ns"] = (
        hit["time"]
    )

    matched_hits.append({

        "event":
            hit["event"],

        "photon_id":
            photon["photon_id"],

        "track_id":
            hit["track_id"],

        "parent_id":
            hit["parent_id"],

        "pmt_id":
            hit["pmt_id"],

        "creation_x_cm":
            photon["creation_x_cm"],

        "creation_y_cm":
            photon["creation_y_cm"],

        "creation_z_cm":
            photon["creation_z_cm"],

        "photon_direction_x":
            photon["direction_x"],

        "photon_direction_y":
            photon["direction_y"],

        "photon_direction_z":
            photon["direction_z"],

        "muon_direction_x":
            photon["muon_direction_x"],

        "muon_direction_y":
            photon["muon_direction_y"],

        "muon_direction_z":
            photon["muon_direction_z"],

        "energy_eV":
            hit["energy"],

        "hit_x_cm":
            hit["x"],

        "hit_y_cm":
            hit["y"],

        "hit_z_cm":
            hit["z"],

        "hit_time_ns":
            hit["time"]
    })


# ============================================================
# MATCHED PHOTONS CSV
# ============================================================

with open(
    "matched_photons.csv",
    "w",
    newline=""
) as f:

    fieldnames = [

        "event",
        "photon_id",
        "track_id",
        "parent_id",
        "pmt_id",

        "creation_x_cm",
        "creation_y_cm",
        "creation_z_cm",

        "photon_direction_x",
        "photon_direction_y",
        "photon_direction_z",

        "muon_direction_x",
        "muon_direction_y",
        "muon_direction_z",

        "energy_eV",

        "hit_x_cm",
        "hit_y_cm",
        "hit_z_cm",

        "hit_time_ns"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        matched_hits
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("SIMULATION COMPLETE")
print("========================================")

print(
    "Events:",
    len(eventAction.event_truth)
)

print(
    "Cherenkov photons:",
    len(eventAction.optical_truth)
)

print(
    "PMT hits:",
    len(detector.pmtSD.all_hits)
)

print(
    "Matched photon hits:",
    len(matched_hits)
)

print(
    "Unmatched PMT hits:",
    len(detector.pmtSD.all_hits)
    -
    len(matched_hits)
)

print()
print("Files:")
print("  event_truth.csv")
print("  photon_truth.csv")
print("  pmt_hits.csv")
print("  matched_photons.csv")

print("========================================")
