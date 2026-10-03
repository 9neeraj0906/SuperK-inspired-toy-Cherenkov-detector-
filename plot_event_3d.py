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

PMT_GEOMETRY_FILE = Path(
    "pmt_geometry.csv"
)

RECONSTRUCTION_FILE = Path(
    "direction_reconstruction_large.csv"
)

EVENT_TRUTH_FILE = Path(
    "event_truth.csv"
)

OUTPUT_FILE = Path(
    "event_3d_display.png"
)


# ============================================================
# DETECTOR GEOMETRY
# ============================================================

WATER_RADIUS_CM = 10.0
WATER_Z_MIN_CM = -5.0
WATER_Z_MAX_CM = 5.0

PMT_RADIUS_CM = 9.5


# ============================================================
# PHYSICS
# ============================================================

N_WATER = 1.336

MUON_MASS_MEV = 105.6583755
MUON_KINETIC_ENERGY_MEV = 400.0

C_CM_NS = 29.9792458


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
# FIXED RECONSTRUCTION VERTEX
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


# ============================================================
# READ DATA
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
                sorted(
                    missing
                )
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


def read_geometry():

    geometry = pd.read_csv(
        PMT_GEOMETRY_FILE
    )

    required = {
        "pmt_id",
        "x_cm",
        "y_cm",
        "z_cm"
    }

    missing = (
        required -
        set(
            geometry.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns in pmt_geometry.csv: "
            +
            str(
                sorted(
                    missing
                )
            )
        )

    return geometry


def read_reconstruction():

    reconstruction = pd.read_csv(
        RECONSTRUCTION_FILE
    )

    required = {
        "event",
        "n_hits",
        "reco_dx",
        "reco_dy",
        "reco_dz",
        "angular_error_deg"
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

    return reconstruction


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

    return truth


# ============================================================
# TRACK EXIT
#
# Used only to draw the muon trajectory inside the water.
# ============================================================

def calculate_exit_distance(
    vertex,
    direction
):

    direction = normalize(
        direction
    )

    ux = direction[0]
    uy = direction[1]
    uz = direction[2]

    # --------------------------------------------------------
    # Bottom surface.
    # --------------------------------------------------------

    if uz < -1.0e-12:

        s_z = (
            WATER_Z_MIN_CM -
            vertex[2]
        ) / uz

    else:

        s_z = np.inf

    # --------------------------------------------------------
    # Top surface.
    # --------------------------------------------------------

    if uz > 1.0e-12:

        s_top = (
            WATER_Z_MAX_CM -
            vertex[2]
        ) / uz

    else:

        s_top = np.inf

    # --------------------------------------------------------
    # Cylindrical wall.
    #
    # Solve:
    #
    # (x0 + s ux)^2 +
    # (y0 + s uy)^2 = R^2
    #
    # Here x0 = y0 = 0.
    # --------------------------------------------------------

    transverse = math.sqrt(
        ux * ux +
        uy * uy
    )

    if transverse > 1.0e-12:

        s_r = (
            WATER_RADIUS_CM /
            transverse
        )

    else:

        s_r = np.inf

    candidates = [
        value
        for value in
        [
            s_z,
            s_top,
            s_r
        ]
        if np.isfinite(value)
        and value > 0.0
    ]

    if not candidates:

        return 0.0

    return min(
        candidates
    )


# ============================================================
# BUILD CONE BASIS
# ============================================================

def build_cone_basis(
    direction
):

    direction = normalize(
        direction
    )

    reference = np.array(
        [
            0.0,
            0.0,
            1.0
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
                1.0,
                0.0,
                0.0
            ],
            dtype=float
        )

    basis_1 = np.cross(
        direction,
        reference
    )

    basis_1 = normalize(
        basis_1
    )

    basis_2 = np.cross(
        direction,
        basis_1
    )

    basis_2 = normalize(
        basis_2
    )

    return (
        basis_1,
        basis_2
    )


# ============================================================
# CONE SURFACE
#
# This is NOT the observed photon pattern.
#
# It is only the ideal Cherenkov cone corresponding to the
# reconstructed direction.
# ============================================================

def make_cone(
    vertex,
    direction,
    length_cm,
    n_length=35,
    n_phi=100
):

    direction = normalize(
        direction
    )

    (
        basis_1,
        basis_2
    ) = build_cone_basis(
        direction
    )

    s_values = np.linspace(
        0.0,
        length_cm,
        n_length
    )

    phi_values = np.linspace(
        0.0,
        2.0 * math.pi,
        n_phi
    )

    cone_x = np.zeros(
        (
            n_length,
            n_phi
        )
    )

    cone_y = np.zeros(
        (
            n_length,
            n_phi
        )
    )

    cone_z = np.zeros(
        (
            n_length,
            n_phi
        )
    )

    for i, s in enumerate(
        s_values
    ):

        center = (
            vertex +
            s *
            direction
        )

        for j, phi in enumerate(
            phi_values
        ):

            transverse = (
                math.cos(phi) *
                basis_1
                +
                math.sin(phi) *
                basis_2
            )

            cone_direction = (
                math.cos(
                    THETA_C_RAD
                )
                *
                direction

                +

                math.sin(
                    THETA_C_RAD
                )
                *
                transverse
            )

            point = (
                center +
                s *
                math.tan(
                    THETA_C_RAD
                )
                *
                transverse
            )

            cone_x[i, j] = point[0]
            cone_y[i, j] = point[1]
            cone_z[i, j] = point[2]

    return (
        cone_x,
        cone_y,
        cone_z
    )


# ============================================================
# CYLINDER WIREFRAME
# ============================================================

def make_cylinder(
    radius,
    z_min,
    z_max,
    n_phi=80
):

    phi = np.linspace(
        0.0,
        2.0 * math.pi,
        n_phi
    )

    z = np.array(
        [
            z_min,
            z_max
        ]
    )

    phi_grid, z_grid = np.meshgrid(
        phi,
        z
    )

    x = (
        radius *
        np.cos(
            phi_grid
        )
    )

    y = (
        radius *
        np.sin(
            phi_grid
        )
    )

    return (
        x,
        y,
        z_grid
    )


# ============================================================
# CHOOSE EVENT
#
# Select the reconstructed event with the largest number of
# detected hits.
# ============================================================

def choose_event(
    reconstruction
):

    valid = (
        reconstruction[
            reconstruction[
                "n_hits"
            ] >= 4
        ]
        .copy()
    )

    if len(valid) == 0:

        raise RuntimeError(
            "No reconstructed event found."
        )

    index = (
        valid[
            "n_hits"
        ]
        .idxmax()
    )

    return (
        valid.loc[
            index
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "3D SUPER-K-LIKE EVENT DISPLAY"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Read files.
    # --------------------------------------------------------

    hits = read_hits()

    geometry = read_geometry()

    reconstruction = (
        read_reconstruction()
    )

    truth = read_truth()

    # --------------------------------------------------------
    # Select event.
    # --------------------------------------------------------

    selected = choose_event(
        reconstruction
    )

    event_id = int(
        selected["event"]
    )

    n_hits = int(
        selected["n_hits"]
    )

    reconstruction_error = float(
        selected[
            "angular_error_deg"
        ]
    )

    reconstructed_direction = normalize(
        np.array(
            [
                selected["reco_dx"],
                selected["reco_dy"],
                selected["reco_dz"]
            ],
            dtype=float
        )
    )

    # --------------------------------------------------------
    # True direction.
    #
    # Used only for validation visualization.
    # --------------------------------------------------------

    truth_event = (
        truth[
            truth["event"] ==
            event_id
        ]
    )

    if len(truth_event) == 0:

        raise RuntimeError(
            f"No truth information for event {event_id}."
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
    # Event hit positions.
    # --------------------------------------------------------

    event_hits = (
        hits[
            hits["event"] ==
            event_id
        ]
        .copy()
    )

    hit_x = (
        event_hits[
            "x_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    hit_y = (
        event_hits[
            "y_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    hit_z = (
        event_hits[
            "z_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # PMT positions.
    # --------------------------------------------------------

    pmt_x = (
        geometry[
            "x_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pmt_y = (
        geometry[
            "y_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    pmt_z = (
        geometry[
            "z_cm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    # --------------------------------------------------------
    # Reconstructed track.
    # --------------------------------------------------------

    reco_length = (
        calculate_exit_distance(
            VERTEX,
            reconstructed_direction
        )
    )

    reco_end = (
        VERTEX +
        reco_length *
        reconstructed_direction
    )

    # --------------------------------------------------------
    # True track.
    #
    # Only for validation/display.
    # --------------------------------------------------------

    true_length = (
        calculate_exit_distance(
            VERTEX,
            true_direction
        )
    )

    true_end = (
        VERTEX +
        true_length *
        true_direction
    )

    # --------------------------------------------------------
    # Idealized reconstructed Cherenkov cone.
    #
    # Limit the displayed cone to the detector region.
    # --------------------------------------------------------

    cone_length = min(
        reco_length,
        12.0
    )

    (
        cone_x,
        cone_y,
        cone_z
    ) = make_cone(
        VERTEX,
        reconstructed_direction,
        cone_length
    )

    # --------------------------------------------------------
    # Detector cylinder.
    # --------------------------------------------------------

    (
        cylinder_x,
        cylinder_y,
        cylinder_z
    ) = make_cylinder(
        WATER_RADIUS_CM,
        WATER_Z_MIN_CM,
        WATER_Z_MAX_CM
    )

    # ========================================================
    # PLOT
    # ========================================================

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Water volume.
    # --------------------------------------------------------

    ax.plot_surface(
        cylinder_x,
        cylinder_y,
        cylinder_z,
        alpha=0.08,
        linewidth=0
    )

    # --------------------------------------------------------
    # PMTs.
    # --------------------------------------------------------

    ax.scatter(
        pmt_x,
        pmt_y,
        pmt_z,
        s=22,
        alpha=0.20,
        label="200 PMTs"
    )

    # --------------------------------------------------------
    # Actual Geant4 photon hits.
    # --------------------------------------------------------

    ax.scatter(
        hit_x,
        hit_y,
        hit_z,
        s=45,
        alpha=0.90,
        label=f"Detected photons ({n_hits})"
    )

    # --------------------------------------------------------
    # Fixed interaction vertex.
    # --------------------------------------------------------

    ax.scatter(
        [VERTEX[0]],
        [VERTEX[1]],
        [VERTEX[2]],
        s=100,
        marker="*",
        label="Interaction vertex"
    )

    # --------------------------------------------------------
    # Reconstructed muon track.
    # --------------------------------------------------------

    ax.plot(
        [
            VERTEX[0],
            reco_end[0]
        ],

        [
            VERTEX[1],
            reco_end[1]
        ],

        [
            VERTEX[2],
            reco_end[2]
        ],

        linewidth=3.0,
        label="Reconstructed muon direction"
    )

    # --------------------------------------------------------
    # True muon track.
    #
    # This is validation only.
    # --------------------------------------------------------

    ax.plot(
        [
            VERTEX[0],
            true_end[0]
        ],

        [
            VERTEX[1],
            true_end[1]
        ],

        [
            VERTEX[2],
            true_end[2]
        ],

        linestyle="--",
        linewidth=2.5,
        label="True muon direction"
    )

    # --------------------------------------------------------
    # Idealized Cherenkov cone.
    #
    # Use a wireframe rather than a solid surface so that it
    # does not hide the actual Geant4 hit pattern.
    # --------------------------------------------------------

    ax.plot_wireframe(
        cone_x,
        cone_y,
        cone_z,
        rstride=3,
        cstride=8,
        alpha=0.18,
        linewidth=0.8,
        label=(
            "Ideal Cherenkov cone "
            f"({THETA_C_DEG:.1f}°)"
        )
    )

    # --------------------------------------------------------
    # Axis labels.
    # --------------------------------------------------------

    ax.set_xlabel(
        "x (cm)",
        labelpad=10
    )

    ax.set_ylabel(
        "y (cm)",
        labelpad=10
    )

    ax.set_zlabel(
        "z (cm)",
        labelpad=10
    )

    # --------------------------------------------------------
    # Equal aspect ratio.
    # --------------------------------------------------------

    ax.set_xlim(
        -11.0,
        11.0
    )

    ax.set_ylim(
        -11.0,
        11.0
    )

    ax.set_zlim(
        -6.0,
        6.0
    )

    try:

        ax.set_box_aspect(
            (
                22,
                22,
                12
            )
        )

    except AttributeError:

        pass

    # --------------------------------------------------------
    # View angle.
    # --------------------------------------------------------

    ax.view_init(
        elev=24,
        azim=38
    )

    # --------------------------------------------------------
    # Title.
    # --------------------------------------------------------

    ax.set_title(
        "Super-K-like Geant4 Cherenkov Event\n"
        f"Event {event_id} | "
        f"{n_hits} detected photons | "
        f"Direction error = "
        f"{reconstruction_error:.2f}°"
    )

    ax.legend(
        loc="upper left",
        fontsize=9
    )

    # --------------------------------------------------------
    # Grid.
    # --------------------------------------------------------

    ax.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    fig.savefig(
        OUTPUT_FILE,
        dpi=250,
        bbox_inches="tight"
    )

    plt.show()

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        f"Event: {event_id}"
    )

    print(
        f"Detected photons: {n_hits}"
    )

    print(
        "Reconstructed direction:",
        "(",
        f"{reconstructed_direction[0]:.5f},",
        f"{reconstructed_direction[1]:.5f},",
        f"{reconstructed_direction[2]:.5f}",
        ")"
    )

    print(
        "True direction:",
        "(",
        f"{true_direction[0]:.5f},",
        f"{true_direction[1]:.5f},",
        f"{true_direction[2]:.5f}",
        ")"
    )

    print(
        f"Direction error: "
        f"{reconstruction_error:.3f} deg"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()
