# SuperK-inspired-toy-Cherenkov-detector-
Geant4 + Python simulation of a Super-Kamiokande-like water Cherenkov detector. Simulates a toy nu_mu CCQE interaction, muon propagation, Cherenkov photon production and PMT detection, followed by Hough-based direction reconstruction, chi2 analysis, and 3D event visualization.


<h1 align="center">Super-Kamiokande-Like Water Cherenkov Detector Simulation</h1>

<p align="center">
  <b>End-to-end Geant4 + Python simulation: ν<sub>μ</sub> interaction → Cherenkov light → PMT hits → muon direction reconstruction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Geant4-11.4.2-blue" alt="Geant4">
  <img src="https://img.shields.io/badge/Python-3.12-yellow" alt="Python">
  <img src="https://img.shields.io/badge/C%2B%2B-pybind11-orange" alt="C++/pybind11">
  <img src="https://img.shields.io/badge/status-toy%20model-lightgrey" alt="Toy model">
</p>

<p align="center">
  <img src="docs/event_565_3d.png" width="600" alt="3D event display: detector, PMT hits, true and reconstructed muon direction, Cherenkov cone">
</p>

> **Scope:** a learning and development project, **not** a realistic Super-Kamiokande simulation. The neutrino interaction is a toy model, and the numbers below describe this simplified setup only.

---

## At a glance

| What I built | How |
|---|---|
| Cylindrical water Cherenkov detector, 200 PMTs | Geant4 via `geant4_pybind` |
| Custom ν<sub>μ</sub> → μ⁻ + p interaction | C++ (`G4VDiscreteProcess`) exposed with `pybind11` |
| Optical photon production and PMT sensitive detector | Geant4 optical physics, custom sensitive detector |
| Photon-to-PMT truth matching | Geant4 track IDs |
| Muon direction reconstruction | Hough-transform search over candidate directions |
| χ² diagnostic and 3D event display | NumPy, Pandas, Matplotlib |

## Key results

| Check | Result |
|---|---|
| Local Cherenkov angle (simulated) | **40.09° ± 0.31°** vs. theory **40.06°** |
| Photon creation → PMT hit matching | Direction agrees within numerical precision |
| Reconstruction sample | 1000 events, 531 with ≥ 4 PMT hits |
| Hough angular error (events ≥ 4 hits) | median **22.6°**, mean 33.7°, best 3.9° |
| Baseline (before Hough step) | mean 40.9° |

Performance depends strongly on how many PMTs fire. Low-hit events reconstruct poorly, as expected.

## Pipeline

```text
νμ ──► toy CCQE ──► μ⁻ + p ──► μ⁻ in water ──► Cherenkov photons ──► PMT hits
                                                                        │
                                              ┌─────────────────────────┴───────┐
                                              ▼                                 ▼
                                   Hough direction fit                   χ² analysis
                                              │
                                              ▼
                                    3D event display
```

---

## Quick start

```bash
# 1. Activate Python environment and Geant4
source <path-to-venv>/bin/activate
source <path-to-geant4-install>/bin/geant4.sh

# 2. Simulate, then analyze
python main.py              # generates pmt_hits.csv, photon truth, etc.
python reconstruct.py       # Hough direction reconstruction
python chi2_analysis.py     # local χ² map
python plot_event_3d.py     # 3D event display
```

Developed with Python 3.12, Geant4 11.4.2, `geant4_pybind` 0.1.3. The custom C++ module is described in [`cpp/README.md`](cpp/README.md).

## Repository layout

```text
detector.py          geometry, materials, PMTs, sensitive detector
Physics.py           physics list: EM + optical + step limiter + custom ν process
Primary.py           primary neutrino generator
Actions.py           Geant4 user actions (event/photon recording)
main.py              run manager and simulation entry point
reconstruct.py       Hough direction reconstruction
chi2_analysis.py     angular χ² map
plot_event_3d.py     3D visualization
cpp/                 custom ν interaction (C++ + pybind11)
data/                example output files
```

---

<details>
<summary><b>Detector geometry</b></summary>

| Component | Radius | Half-height | Material |
|---|---|---|---|
| Outer detector | 20 cm | 10 cm | `G4_WATER` |
| Inner detector | 10 cm | 5 cm | `G4_WATER` |

- 200 PMTs: 40 per ring × 5 rings, on the inner detector surface
- PMT radius 0.5 cm, radial position 9.5 cm
- Positions stored in `pmt_geometry.csv`

</details>

<details>
<summary><b>Toy neutrino interaction</b></summary>

ν<sub>μ</sub> → μ⁻ + p, with fixed kinetic energies (T<sub>μ</sub> = 400 MeV, T<sub>p</sub> = 100 MeV). Both secondaries inherit the neutrino direction, and the interaction is forced to occur.

Not modelled: cross sections, realistic CCQE kinematics, nuclear effects, Fermi motion, binding energy, final-state angular distributions, energy-dependent rates, oscillations.

</details>

<details>
<summary><b>Photon truth matching</b></summary>

Cherenkov photon creation records (position, direction, energy, parent muon direction) are linked to PMT hits through optical-photon track IDs. The photon direction was compared with the vector from creation point to PMT hit; the difference is zero within numerical precision, confirming the records associate correctly.

</details>

<details>
<summary><b>χ² analysis (Event 565)</b></summary>

Simplified angular diagnostic:

$$\chi^2 = \sum_i \frac{(\theta_i - \theta_C)^2}{\sigma_\theta^2}$$

where θ<sub>i</sub> is the angle between a candidate direction and PMT *i*, θ<sub>C</sub> = 40.06° is the expected Cherenkov angle, and σ<sub>θ</sub> = 4° is an assumed angular uncertainty.

Event 565 is a high-multiplicity event (256 hits, far above the sample mean of 20), chosen for display.

| Quantity | Value |
|---|---|
| χ²<sub>min</sub> | 3690.97 |
| χ²(true direction) | 3820.38 (Δχ² = 129.4) |
| χ²(Hough direction) | 4033.06 (Δχ² = 342.1) |
| χ² minimum vs. truth | 3.93° apart |
| Hough vs. truth | 6.73° apart |

These Δχ² values are not calibrated confidence levels. The large absolute χ² reflects the simplifications: fixed vertex, point-source approximation, simplified PMT response, assumed σ<sub>θ</sub>.

</details>

<details>
<summary><b>3D event display</b></summary>

Shows the detector, PMT positions, Geant4 PMT hits, true and reconstructed muon directions, and the ideal Cherenkov cone around the reconstructed direction. The cone is the expected geometry, not a rendering of individual photon trajectories.

</details>

---

## Limitations

- **Neutrino interaction:** toy model, fixed energies, forced interaction. Not a physical event generator.
- **Geometry:** a small cylinder, not the real Super-Kamiokande geometry.
- **PMTs:** ideal sensitive surfaces. No quantum efficiency, gain, dark noise, afterpulsing, or calibration.
- **Reconstruction:** simplified Hough search, not fiTQun or any full likelihood reconstruction.
- **χ²:** a diagnostic only, not a statistical test.

## Skills demonstrated

**Geant4** detector simulation and optical physics · **C++/Python** integration (`pybind11`) · Cherenkov radiation · PMT sensitive detectors · Monte Carlo event data · photon truth matching · Hough-based direction reconstruction · χ² analysis · 3D event visualization · validation against analytic expectations (θ<sub>C</sub>)

---

<p align="center"><i>Author: <b>Your Name</b> · <a href="mailto:you@email.com">you@email.com</a> · <a href="https://github.com/your-username">GitHub</a></i></p>
