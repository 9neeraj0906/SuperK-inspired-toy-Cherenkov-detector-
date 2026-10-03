from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# FILES
# ============================================================

PMT_HITS_FILE = Path("pmt_hits.csv")
EVENT_TRUTH_FILE = Path("event_truth.csv")

OUTPUT_FILE = Path(
    "direction_reconstruction_large.csv"
)


# ============================================================
# FIXED VERTEX
# ============================================================

VERTEX = np.array(
    [0.0, 0.0, 5.0],
    dtype=float
)


# ============================================================
# PHYSICS
# ============================================================

N_WATER = 1.336

MUON_MASS_MEV = 105.6583755
MUON_KINETIC_ENERGY_MEV = 400.0


def calculate_beta():

    kinetic = MUON_KINETIC_ENERGY_MEV
    mass = MUON_MASS_MEV

    momentum = math.sqrt(
        kinetic *
        (
            kinetic +
            2.0 * mass
        )
    )

    total_energy = (
        kinetic +
        mass
    )

    return (
        momentum /
        total_energy
    )


BETA = calculate_beta()


# ============================================================
# CHERENKOV ANGLE
# ============================================================

THETA_C_RAD = math.acos(
    1.0 /
    (
        BETA *
        N_WATER
    )
)

THETA_C_DEG = math.degrees(
    THETA_C_RAD
)


# ============================================================
# HOUGH PARAMETERS
# ============================================================

N_HOUGH_DIRECTIONS = 30000

HOUGH_SIGMA_DEG = 4.0

HOUGH_SIGMA_RAD = math.radians(
    HOUGH_SIGMA_DEG
)

H0UGH_CHUNK_SIZE = 2000

MIN_HITS = 4


# ============================================================
# HIT WEIGHTING
# ============================================================

L_ATTENUATION_CM = 4000.0


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=float
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1.0e-15:

        return np.array(
            [0.0, 0.0, -1.0],
            dtype=float
        )

    return (
        vector /
        norm
    )


def angular_error_deg(
    direction_a,
    direction_b
):

    a = normalize(
        direction_a
    )

    b = normalize(
        direction_b
    )

    cosine = np.dot(
        a,
        b
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(
            cosine
        )
    )


# ============================================================
# FIBONACCI SPHERE
#
# Same full-sphere grid as the current 20.34 degree
# actual-hit baseline.
# ============================================================

def fibonacci_sphere(n):

    index = (
        np.arange(n)
        + 0.5
    )

    z = (
        1.0 -
        2.0 *
        index /
        n
    )

    radius = np.sqrt(
        np.maximum(
            0.0,
            1.0 -
            z * z
        )
    )

    golden_angle = (
        math.pi *
        (
            3.0 -
            math.sqrt(5.0)
        )
    )

    phi = (
        golden_angle *
        index
    )

    x = (
        radius *
        np.cos(phi)
    )

    y = (
        radius *
        np.sin(phi)
    )

    return np.column_stack(
        (
            x,
            y,
            z
        )
    )


# ============================================================
# READ PMT HITS
# ============================================================

def read_hits():

    hits = pd.read_csv(
        PMT_HITS_FILE
    )

    required = {
        "event",
        "pmt_id",
        "x_cm",
        "y_cm",
        "z_cm"
    }

    missing = (
        required -
        set(
            hits.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns in pmt_hits.csv: "
            +
            str(
                sorted(missing)
            )
        )

    hits["event"] = (
        hits["event"]
        .astype(int)
    )

    hits["pmt_id"] = (
        hits["pmt_id"]
        .astype(int)
    )

    hits["x_cm"] = (
        hits["x_cm"]
        .astype(float)
    )

    hits["y_cm"] = (
        hits["y_cm"]
        .astype(float)
    )

    hits["z_cm"] = (
        hits["z_cm"]
        .astype(float)
    )

    return hits


# ============================================================
# READ TRUTH
# ============================================================

def read_truth():

    truth = pd.read_csv(
        EVENT_TRUTH_FILE
    )

    required = {
        "event",
        "initial_dx",
        "initial_dy",
        "initial_dz"
    }

    missing = (
        required -
        set(
            truth.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns in event_truth.csv: "
            +
            str(
                sorted(missing)
            )
        )

    truth["event"] = (
        truth["event"]
        .astype(int)
    )

    return truth


# ============================================================
# HIT GEOMETRY
# ============================================================

def calculate_hit_geometry(
    positions
):

    vectors = (
        positions -
        VERTEX[None, :]
    )

    distances = np.linalg.norm(
        vectors,
        axis=1
    )

    safe_distances = np.maximum(
        distances,
        1.0e-12
    )

    directions = (
        vectors /
        safe_distances[:, None]
    )

    return (
        directions,
        distances
    )


# ============================================================
# HIT WEIGHTS
# ============================================================

def calculate_hit_weights(
    positions,
    distances,
    hit_directions
):

    radial = np.column_stack(
        (
            positions[:, 0],
            positions[:, 1],
            np.zeros(
                len(positions)
            )
        )
    )

    radial_norm = np.linalg.norm(
        radial,
        axis=1
    )

    radial_norm = np.maximum(
        radial_norm,
        1.0e-12
    )

    radial = (
        radial /
        radial_norm[:, None]
    )

    cos_incidence = np.sum(
        hit_directions *
        radial,
        axis=1
    )

    cos_incidence = np.clip(
        cos_incidence,
        0.0,
        1.0
    )

    attenuation = np.exp(
        distances /
        L_ATTENUATION_CM
    )

    weights = (
        attenuation *
        cos_incidence
    )

    weights = np.maximum(
        weights,
        1.0e-6
    )

    return weights


# ============================================================
# INITIAL VECTOR ESTIMATE
# ============================================================

def calculate_initial_direction(
    hit_directions,
    hit_weights
):

    vector = np.sum(
        hit_weights[:, None] *
        hit_directions,
        axis=0
    )

    return normalize(
        vector
    )


# ============================================================
# HOUGH RECONSTRUCTION
#
# Chunked over directions so the 30,000-direction search
# does not create a huge temporary matrix.
# ============================================================

def reconstruct_hough(
    hit_directions,
    hit_weights,
    candidate_directions
):

    best_score = -np.inf

    best_index = -1

    n_candidates = (
        len(
            candidate_directions
        )
    )

    for start in range(
        0,
        n_candidates,
        H0UGH_CHUNK_SIZE
    ):

        stop = min(
            start +
            H0UGH_CHUNK_SIZE,
            n_candidates
        )

        candidates = (
            candidate_directions[
                start:stop
            ]
        )

        cosine = (
            candidates @
            hit_directions.T
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        angles = np.arccos(
            cosine
        )

        residual = (
            angles -
            THETA_C_RAD
        )

        response = np.exp(
            -0.5 *
            (
                residual /
                HOUGH_SIGMA_RAD
            ) ** 2
        )

        scores = np.sum(
            response *
            hit_weights[None, :],
            axis=1
        )

        local_index = int(
            np.argmax(
                scores
            )
        )

        local_score = float(
            scores[
                local_index
            ]
        )

        if local_score > best_score:

            best_score = local_score

            best_index = (
                start +
                local_index
            )

    return (
        candidate_directions[
            best_index
        ].copy(),

        float(
            best_score
        )
    )


# ============================================================
# PMT-HIT COUNT BIN
# ============================================================

def hit_bin(
    n_hits
):

    if n_hits < 4:
        return "<4"

    if n_hits < 6:
        return "4-5"

    if n_hits < 10:
        return "6-9"

    if n_hits < 20:
        return "10-19"

    if n_hits < 50:
        return "20-49"

    if n_hits < 100:
        return "50-99"

    return "100+"


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "LARGE-SAMPLE CHERENKOV DIRECTION RECONSTRUCTION"
    )
    print("=" * 60)

    print(
        f"Fixed vertex: "
        f"({VERTEX[0]:.1f}, "
        f"{VERTEX[1]:.1f}, "
        f"{VERTEX[2]:.1f}) cm"
    )

    print(
        f"Muon beta: "
        f"{BETA:.6f}"
    )

    print(
        f"Cherenkov angle: "
        f"{THETA_C_DEG:.4f} deg"
    )

    print(
        f"Hough directions: "
        f"{N_HOUGH_DIRECTIONS}"
    )

    # --------------------------------------------------------
    # Read data.
    # --------------------------------------------------------

    hits = read_hits()

    truth = read_truth()

    print(
        f"Total PMT hit rows: "
        f"{len(hits)}"
    )

    print(
        f"Truth events: "
        f"{len(truth)}"
    )

    # --------------------------------------------------------
    # Candidate directions.
    # --------------------------------------------------------

    candidate_directions = (
        fibonacci_sphere(
            N_HOUGH_DIRECTIONS
        )
    )

    # --------------------------------------------------------
    # Truth lookup.
    # --------------------------------------------------------

    truth_lookup = {
        int(row["event"]): row
        for _, row in
        truth.iterrows()
    }

    # --------------------------------------------------------
    # Process every truth event.
    # --------------------------------------------------------

    event_results = []

    truth_events = sorted(
        truth["event"]
        .unique()
    )

    total_events = len(
        truth_events
    )

    print()
    print(
        "Running reconstruction..."
    )

    for counter, event_id in enumerate(
        truth_events,
        start=1
    ):

        event_hits = (
            hits[
                hits["event"] ==
                event_id
            ]
        )

        positions = (
            event_hits[
                [
                    "x_cm",
                    "y_cm",
                    "z_cm"
                ]
            ]
            .to_numpy(
                dtype=float
            )
        )

        n_hits = len(
            positions
        )

        truth_row = (
            truth_lookup[
                event_id
            ]
        )

        true_direction = normalize(
            np.array(
                [
                    float(
                        truth_row[
                            "initial_dx"
                        ]
                    ),

                    float(
                        truth_row[
                            "initial_dy"
                        ]
                    ),

                    float(
                        truth_row[
                            "initial_dz"
                        ]
                    )
                ],
                dtype=float
            )
        )

        # ----------------------------------------------------
        # Insufficient events.
        # ----------------------------------------------------

        if n_hits < MIN_HITS:

            event_results.append(
                {
                    "event":
                        event_id,

                    "n_hits":
                        n_hits,

                    "hit_bin":
                        hit_bin(
                            n_hits
                        ),

                    "reconstructed":
                        False,

                    "initial_dx":
                        np.nan,

                    "initial_dy":
                        np.nan,

                    "initial_dz":
                        np.nan,

                    "reco_dx":
                        np.nan,

                    "reco_dy":
                        np.nan,

                    "reco_dz":
                        np.nan,

                    "initial_error_deg":
                        np.nan,

                    "angular_error_deg":
                        np.nan,

                    "hough_score":
                        np.nan
                }
            )

            continue

        # ----------------------------------------------------
        # Hit directions from fixed vertex.
        # ----------------------------------------------------

        (
            hit_directions,
            distances
        ) = calculate_hit_geometry(
            positions
        )

        # ----------------------------------------------------
        # Corrected weights.
        # ----------------------------------------------------

        hit_weights = (
            calculate_hit_weights(
                positions,
                distances,
                hit_directions
            )
        )

        # ----------------------------------------------------
        # Initial vector estimate.
        # ----------------------------------------------------

        initial = (
            calculate_initial_direction(
                hit_directions,
                hit_weights
            )
        )

        # ----------------------------------------------------
        # Hough reconstruction.
        # ----------------------------------------------------

        (
            reconstructed,
            hough_score
        ) = reconstruct_hough(
            hit_directions,
            hit_weights,
            candidate_directions
        )

        # ----------------------------------------------------
        # Validation against truth.
        # ----------------------------------------------------

        initial_error = (
            angular_error_deg(
                initial,
                true_direction
            )
        )

        reconstruction_error = (
            angular_error_deg(
                reconstructed,
                true_direction
            )
        )

        event_results.append(
            {
                "event":
                    event_id,

                "n_hits":
                    n_hits,

                "hit_bin":
                    hit_bin(
                        n_hits
                    ),

                "reconstructed":
                    True,

                "initial_dx":
                    initial[0],

                "initial_dy":
                    initial[1],

                "initial_dz":
                    initial[2],

                "reco_dx":
                    reconstructed[0],

                "reco_dy":
                    reconstructed[1],

                "reco_dz":
                    reconstructed[2],

                "initial_error_deg":
                    initial_error,

                "angular_error_deg":
                    reconstruction_error,

                "hough_score":
                    hough_score
            }
        )

        # ----------------------------------------------------
        # Very small progress output.
        # ----------------------------------------------------

        if (
            counter % 100 == 0
            or
            counter == total_events
        ):

            print(
                f"Processed "
                f"{counter}/{total_events}"
            )

    # ========================================================
    # SAVE EVENT RESULTS
    # ========================================================

    output = pd.DataFrame(
        event_results
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # RECONSTRUCTED EVENTS
    # ========================================================

    reconstructed = (
        output[
            output[
                "reconstructed"
            ]
            == True
        ]
        .copy()
    )

    if len(
        reconstructed
    ) == 0:

        print()
        print(
            "No events with enough hits."
        )

        return

    errors = (
        reconstructed[
            "angular_error_deg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    initial_errors = (
        reconstructed[
            "initial_error_deg"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    print(
        f"Total truth events: "
        f"{len(output)}"
    )

    print(
        f"Reconstructed events "
        f"(>= {MIN_HITS} hits): "
        f"{len(reconstructed)}"
    )

    print(
        f"Events below threshold: "
        f"{len(output) - len(reconstructed)}"
    )

    print()

    print(
        f"Initial mean error: "
        f"{np.mean(initial_errors):.4f} deg"
    )

    print(
        f"Hough mean error: "
        f"{np.mean(errors):.4f} deg"
    )

    print(
        f"Hough median error: "
        f"{np.median(errors):.4f} deg"
    )

    print(
        f"Hough RMS error: "
        f"{math.sqrt(np.mean(errors ** 2)):.4f} deg"
    )

    print(
        f"Best error: "
        f"{np.min(errors):.4f} deg"
    )

    print(
        f"Worst error: "
        f"{np.max(errors):.4f} deg"
    )

    # ========================================================
    # HIT COUNT STATISTICS
    # ========================================================

    print()
    print("=" * 60)
    print("ERROR VS DETECTED-HIT COUNT")
    print("=" * 60)

    bins = [
        "4-5",
        "6-9",
        "10-19",
        "20-49",
        "50-99",
        "100+"
    ]

    for current_bin in bins:

        subset = (
            reconstructed[
                reconstructed[
                    "hit_bin"
                ]
                == current_bin
            ]
        )

        if len(subset) == 0:

            continue

        subset_errors = (
            subset[
                "angular_error_deg"
            ]
            .to_numpy(
                dtype=float
            )
        )

        print(
            f"{current_bin:>6s} hits | "
            f"N = {len(subset):4d} | "
            f"mean = "
            f"{np.mean(subset_errors):7.3f}° | "
            f"median = "
            f"{np.median(subset_errors):7.3f}°"
        )

    # ========================================================
    # MEAN / MEDIAN HIT COUNT
    # ========================================================

    hit_counts = (
        reconstructed[
            "n_hits"
        ]
        .to_numpy(
            dtype=int
        )
    )

    print()
    print(
        f"Mean detected hits/event: "
        f"{np.mean(hit_counts):.2f}"
    )

    print(
        f"Median detected hits/event: "
        f"{np.median(hit_counts):.0f}"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
