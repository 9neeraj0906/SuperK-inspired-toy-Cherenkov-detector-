from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

PMT_HITS_FILE = Path(
    "pmt_hits.csv"
)

EVENT_TRUTH_FILE = Path(
    "event_truth.csv"
)

RECONSTRUCTION_FILE = Path(
    "direction_reconstruction_large.csv"
)

OUTPUT_CSV = Path(
    "chi2_local_event_565.csv"
)

OUTPUT_PLOT = Path(
    "chi2_local_map_event_565.png"
)


# ============================================================
# EVENT
# ============================================================

EVENT_ID = 565


# ============================================================
# FIXED VERTEX
# ============================================================

VERTEX = np.array(
    [
        0.0,
        0.0,
        5.0
    ],
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
# CHI2 UNCERTAINTY
# ============================================================

SIGMA_DEG = 4.0

SIGMA_RAD = math.radians(
    SIGMA_DEG
)


# ============================================================
# LOCAL MAP
# ============================================================

MAP_LIMIT_DEG = 12.0

MAP_STEP_DEG = 0.10


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize(
    vector
):

    vector = np.asarray(
        vector,
        dtype=float
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1.0e-15:

        raise ValueError(
            "Cannot normalize zero vector."
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
# READ PMT HITS
# ============================================================

def read_hits():

    hits = pd.read_csv(
        PMT_HITS_FILE
    )

    required = {
        "event",
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
                sorted(
                    missing
                )
            )
        )

    hits["event"] = (
        hits["event"]
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
                sorted(
                    missing
                )
            )
        )

    truth["event"] = (
        truth["event"]
        .astype(int)
    )

    return truth


# ============================================================
# READ RECONSTRUCTION
# ============================================================

def read_reconstruction():

    reconstruction = pd.read_csv(
        RECONSTRUCTION_FILE
    )

    required = {
        "event",
        "reco_dx",
        "reco_dy",
        "reco_dz"
    }

    missing = (
        required -
        set(
            reconstruction.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns in "
            "direction_reconstruction_large.csv: "
            +
            str(
                sorted(
                    missing
                )
            )
        )

    reconstruction["event"] = (
        reconstruction["event"]
        .astype(int)
    )

    return reconstruction


# ============================================================
# BUILD HIT DIRECTIONS
# ============================================================

def calculate_hit_directions(
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

    return directions


# ============================================================
# CHI2
# ============================================================

def calculate_chi2_for_direction(
    direction,
    hit_directions
):

    direction = normalize(
        direction
    )

    cosine = (
        hit_directions @
        direction
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

    chi2 = np.sum(
        (
            residual /
            SIGMA_RAD
        ) ** 2
    )

    return float(
        chi2
    )


# ============================================================
# LOCAL TANGENT BASIS
#
# e1 and e2 are perpendicular to the true direction.
# The map coordinates are angular offsets from truth.
# ============================================================

def build_tangent_basis(
    direction
):

    direction = normalize(
        direction
    )

    reference = np.array(
        [
            1.0,
            0.0,
            0.0
        ],
        dtype=float
    )

    if abs(
        np.dot(
            direction,
            reference
        )
    ) > 0.9:

        reference = np.array(
            [
                0.0,
                1.0,
                0.0
            ],
            dtype=float
        )

    e1 = (
        reference -
        np.dot(
            reference,
            direction
        )
        *
        direction
    )

    e1 = normalize(
        e1
    )

    e2 = np.cross(
        direction,
        e1
    )

    e2 = normalize(
        e2
    )

    return (
        e1,
        e2
    )


# ============================================================
# DIRECTION FROM LOCAL ANGULAR COORDINATES
#
# dx_deg and dy_deg are angular offsets on the tangent plane.
#
# r = sqrt(dx^2 + dy^2)
#
# u =
#     cos(r) * truth
#     +
#     sin(r) * tangent_direction
# ============================================================

def direction_from_local_angles(
    dx_deg,
    dy_deg,
    center_direction,
    basis_1,
    basis_2
):

    radius_deg = math.sqrt(
        dx_deg * dx_deg +
        dy_deg * dy_deg
    )

    radius_rad = math.radians(
        radius_deg
    )

    if radius_deg <= 1.0e-12:

        return center_direction.copy()

    tangent = (
        dx_deg *
        basis_1
        +
        dy_deg *
        basis_2
    )

    tangent = normalize(
        tangent
    )

    direction = (
        math.cos(
            radius_rad
        )
        *
        center_direction

        +

        math.sin(
            radius_rad
        )
        *
        tangent
    )

    return normalize(
        direction
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "LOCAL CHERENKOV CHI2 MAP"
    )
    print("=" * 60)

    print(
        f"Event: {EVENT_ID}"
    )

    print(
        f"Fixed vertex: "
        f"({VERTEX[0]:.1f}, "
        f"{VERTEX[1]:.1f}, "
        f"{VERTEX[2]:.1f}) cm"
    )

    print(
        f"Cherenkov angle: "
        f"{THETA_C_DEG:.4f} deg"
    )

    print(
        f"Angular sigma: "
        f"{SIGMA_DEG:.2f} deg"
    )

    print(
        f"Map range: "
        f"±{MAP_LIMIT_DEG:.1f} deg"
    )

    print(
        f"Map step: "
        f"{MAP_STEP_DEG:.2f} deg"
    )

    # --------------------------------------------------------
    # Read files.
    # --------------------------------------------------------

    hits = read_hits()

    truth = read_truth()

    reconstruction = (
        read_reconstruction()
    )

    # --------------------------------------------------------
    # Select event hits.
    # --------------------------------------------------------

    event_hits = (
        hits[
            hits["event"] ==
            EVENT_ID
        ]
        .copy()
    )

    if len(event_hits) == 0:

        raise RuntimeError(
            f"Event {EVENT_ID} not found."
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

    print(
        f"Detected hits: "
        f"{n_hits}"
    )

    # --------------------------------------------------------
    # Convert hits to directions from fixed vertex.
    # --------------------------------------------------------

    hit_directions = (
        calculate_hit_directions(
            positions
        )
    )

    # --------------------------------------------------------
    # True direction.
    #
    # Used as the center of the visualization.
    # --------------------------------------------------------

    truth_event = (
        truth[
            truth["event"] ==
            EVENT_ID
        ]
    )

    if len(truth_event) == 0:

        raise RuntimeError(
            f"Truth for event {EVENT_ID} not found."
        )

    truth_row = (
        truth_event.iloc[0]
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

    # --------------------------------------------------------
    # Reconstructed Hough direction.
    # --------------------------------------------------------

    reconstruction_event = (
        reconstruction[
            reconstruction["event"] ==
            EVENT_ID
        ]
    )

    if len(
        reconstruction_event
    ) == 0:

        raise RuntimeError(
            f"Reconstruction for event "
            f"{EVENT_ID} not found."
        )

    reconstruction_row = (
        reconstruction_event.iloc[0]
    )

    hough_direction = normalize(
        np.array(
            [
                float(
                    reconstruction_row[
                        "reco_dx"
                    ]
                ),

                float(
                    reconstruction_row[
                        "reco_dy"
                    ]
                ),

                float(
                    reconstruction_row[
                        "reco_dz"
                    ]
                )
            ],
            dtype=float
        )
    )

    hough_error = (
        angular_error_deg(
            hough_direction,
            true_direction
        )
    )

    # --------------------------------------------------------
    # Tangent-plane basis centered on TRUE direction.
    # --------------------------------------------------------

    (
        basis_1,
        basis_2
    ) = build_tangent_basis(
        true_direction
    )

    # ========================================================
    # BUILD LOCAL GRID
    # ========================================================

    axis = np.arange(
        -MAP_LIMIT_DEG,
        MAP_LIMIT_DEG +
        MAP_STEP_DEG * 0.5,
        MAP_STEP_DEG
    )

    dx_grid, dy_grid = np.meshgrid(
        axis,
        axis
    )

    chi2_grid = np.empty(
        dx_grid.shape,
        dtype=float
    )

    # ========================================================
    # CALCULATE LOCAL CHI2
    # ========================================================

    print()
    print(
        "Calculating local chi2 map..."
    )

    for i in range(
        dx_grid.shape[0]
    ):

        for j in range(
            dx_grid.shape[1]
        ):

            direction = (
                direction_from_local_angles(
                    dx_grid[i, j],
                    dy_grid[i, j],
                    true_direction,
                    basis_1,
                    basis_2
                )
            )

            chi2_grid[i, j] = (
                calculate_chi2_for_direction(
                    direction,
                    hit_directions
                )
            )

    # ========================================================
    # FIND LOCAL MINIMUM
    # ========================================================

    minimum_index = np.unravel_index(
        np.argmin(
            chi2_grid
        ),
        chi2_grid.shape
    )

    minimum_i = (
        minimum_index[0]
    )

    minimum_j = (
        minimum_index[1]
    )

    minimum_dx = float(
        dx_grid[
            minimum_i,
            minimum_j
        ]
    )

    minimum_dy = float(
        dy_grid[
            minimum_i,
            minimum_j
        ]
    )

    minimum_direction = (
        direction_from_local_angles(
            minimum_dx,
            minimum_dy,
            true_direction,
            basis_1,
            basis_2
        )
    )

    minimum_chi2 = float(
        chi2_grid[
            minimum_i,
            minimum_j
        ]
    )

    true_chi2 = (
        calculate_chi2_for_direction(
            true_direction,
            hit_directions
        )
    )

    hough_chi2 = (
        calculate_chi2_for_direction(
            hough_direction,
            hit_directions
        )
    )

    delta_chi2 = (
        chi2_grid -
        minimum_chi2
    )

    hough_vs_minimum = (
        angular_error_deg(
            hough_direction,
            minimum_direction
        )
    )

    minimum_vs_truth = (
        angular_error_deg(
            minimum_direction,
            true_direction
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("LOCAL CHI2 RESULTS")
    print("=" * 60)

    print(
        f"Local minimum chi2: "
        f"{minimum_chi2:.4f}"
    )

    print(
        f"True-direction chi2: "
        f"{true_chi2:.4f}"
    )

    print(
        f"Hough-direction chi2: "
        f"{hough_chi2:.4f}"
    )

    print()

    print(
        "Chi2 minimum offset from truth:"
    )

    print(
        f"  dx = "
        f"{minimum_dx:.3f} deg"
    )

    print(
        f"  dy = "
        f"{minimum_dy:.3f} deg"
    )

    print(
        f"  angular distance = "
        f"{minimum_vs_truth:.3f} deg"
    )

    print()

    print(
        f"Hough error vs truth: "
        f"{hough_error:.3f} deg"
    )

    print(
        f"Hough vs chi2 minimum: "
        f"{hough_vs_minimum:.3f} deg"
    )

    print(
        f"True Δchi2: "
        f"{true_chi2 - minimum_chi2:.4f}"
    )

    print(
        f"Hough Δchi2: "
        f"{hough_chi2 - minimum_chi2:.4f}"
    )

    # ========================================================
    # SAVE LOCAL MAP
    # ========================================================

    local_output = pd.DataFrame(
        {
            "dx_deg": dx_grid.ravel(),

            "dy_deg": dy_grid.ravel(),

            "chi2": chi2_grid.ravel(),

            "delta_chi2":
                delta_chi2.ravel()
        }
    )

    local_output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    # --------------------------------------------------------
    # Filled chi2 map.
    # --------------------------------------------------------

    image = ax.pcolormesh(
        dx_grid,
        dy_grid,
        delta_chi2,
        shading="auto"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        r"$\Delta\chi^2$"
    )

    # --------------------------------------------------------
    # Optional contours around the minimum.
    #
    # These are shown only as Δχ² levels; they are not being
    # interpreted as calibrated confidence regions.
    # --------------------------------------------------------

    contour_levels = [
        2.30,
        4.61,
        9.21
    ]

    contour = ax.contour(
        dx_grid,
        dy_grid,
        delta_chi2,
        levels=contour_levels,
        linewidths=1.2
    )

    ax.clabel(
        contour,
        inline=True,
        fontsize=9,
        fmt="Δχ² = %.2f"
    )

    # --------------------------------------------------------
    # TRUE direction.
    #
    # By construction this is (0,0).
    # --------------------------------------------------------

    ax.scatter(
        [0.0],
        [0.0],
        marker="o",
        s=130,
        facecolors="none",
        linewidths=2.5,
        label="True direction"
    )

    # --------------------------------------------------------
    # Local chi2 minimum.
    # --------------------------------------------------------

    ax.scatter(
        [minimum_dx],
        [minimum_dy],
        marker="*",
        s=250,
        label="χ² minimum"
    )

    # --------------------------------------------------------
    # Hough reconstruction projected into tangent plane.
    # --------------------------------------------------------

    # Angular displacement from truth.
    #
    # Projection:
    #
    # alpha = atan2(
    #     u . e1,
    #     u . truth
    # )
    #
    # beta  = atan2(
    #     u . e2,
    #     u . truth
    # )

    hough_parallel = np.dot(
        hough_direction,
        true_direction
    )

    hough_e1 = np.dot(
        hough_direction,
        basis_1
    )

    hough_e2 = np.dot(
        hough_direction,
        basis_2
    )

    hough_alpha = math.degrees(
        math.atan2(
            hough_e1,
            hough_parallel
        )
    )

    hough_beta = math.degrees(
        math.atan2(
            hough_e2,
            hough_parallel
        )
    )

    ax.scatter(
        [hough_alpha],
        [hough_beta],
        marker="x",
        s=170,
        linewidths=3,
        label="Hough reconstruction"
    )

    # --------------------------------------------------------
    # Formatting.
    # --------------------------------------------------------

    ax.set_xlim(
        -MAP_LIMIT_DEG,
        MAP_LIMIT_DEG
    )

    ax.set_ylim(
        -MAP_LIMIT_DEG,
        MAP_LIMIT_DEG
    )

    ax.set_xlabel(
        "Angular offset Δα from true direction (degrees)"
    )

    ax.set_ylabel(
        "Angular offset Δβ from true direction (degrees)"
    )

    ax.set_title(
        "Event 565 Local Cherenkov χ² Landscape\n"
        f"{n_hits} detected photons | "
        f"θC = {THETA_C_DEG:.2f}°"
    )

    ax.axhline(
        0.0,
        linewidth=0.7,
        alpha=0.4
    )

    ax.axvline(
        0.0,
        linewidth=0.7,
        alpha=0.4
    )

    ax.grid(
        alpha=0.20
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.legend(
        loc="upper right"
    )

    # --------------------------------------------------------
    # Information box.
    # --------------------------------------------------------

    information = (
        f"χ²min = {minimum_chi2:.2f}\n"
        f"χ²true = {true_chi2:.2f}\n"
        f"χ²Hough = {hough_chi2:.2f}\n"
        f"Hough error = {hough_error:.2f}°"
    )

    ax.text(
        0.02,
        0.02,
        information,
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85
        )
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PLOT,
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # SAVE / FINISH
    # ========================================================

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print(
        f"Numerical map: "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Plot: "
        f"{OUTPUT_PLOT}"
    )

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":

    main()
