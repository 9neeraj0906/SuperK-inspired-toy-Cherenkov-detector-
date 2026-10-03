from geant4_pybind import *


# ============================================================
# Event Action
# ============================================================

class EventAction(G4UserEventAction):

    def __init__(self, pmtSD):

        super().__init__()

        self.pmtSD = pmtSD

        # Event-level truth
        self.event_truth = []

        # Optical photon truth
        self.optical_truth = []

        # Direct mapping:
        # (event_id, track_id) -> photon_id
        self.track_to_photon = {}

        # Photon counter
        self.photon_counter = 0

        # Current muon truth
        self.muon_truth = None

        # Current event
        self.current_event = None

        # ----------------------------------------------------
        # Temporary Cherenkov creation information.
        #
        # These records are created when the muon produces
        # Cherenkov photons. They are transferred to the actual
        # Geant4 optical tracks in TrackingAction.
        # ----------------------------------------------------
        self.pending_muon_states = []


    # ========================================================
    # Begin event
    # ========================================================

    def BeginOfEventAction(self, event):

        event_id = event.GetEventID()

        self.current_event = event_id

        self.muon_truth = None

        self.photon_counter = 0

        self.track_to_photon = {}

        self.pending_muon_states = []

        print(
            "Event",
            event_id,
            "| started"
        )


    # ========================================================
    # End event
    # ========================================================

    def EndOfEventAction(self, event):

        event_id = event.GetEventID()

        # ----------------------------------------------------
        # Save muon truth
        # ----------------------------------------------------

        if self.muon_truth is not None:

            truth = self.muon_truth.copy()

            truth["event"] = event_id

            self.event_truth.append(
                truth
            )

        # ----------------------------------------------------
        # Count photons belonging to this event
        # ----------------------------------------------------

        event_photons = 0

        for photon in self.optical_truth:

            if photon["event"] == event_id:

                event_photons += 1

        # ----------------------------------------------------
        # Event summary
        # ----------------------------------------------------

        print(
            "Event",
            event_id,
            "| photons =",
            event_photons,
            "| PMT hits =",
            self.pmtSD.hit_count
        )


# ============================================================
# Tracking Action
#
# The optical photon information is taken directly from the
# actual G4Track in PreUserTrackingAction.
#
# The photon-to-PMT association remains based directly on
# the real Geant4 optical track ID.
# ============================================================

class TrackingAction(G4UserTrackingAction):

    def __init__(self, eventAction):

        super().__init__()

        self.eventAction = eventAction


    # ========================================================
    # Find local muon state belonging to an optical photon
    # ========================================================

    def FindMuonState(
        self,
        vertex,
        direction,
        energy
    ):

        if not self.eventAction.pending_muon_states:

            return None

        best_state = None

        best_score = float("inf")

        for state in self.eventAction.pending_muon_states:

            # ------------------------------------------------
            # Position difference in cm
            # ------------------------------------------------

            dx = (
                state["creation_x_cm"]
                -
                vertex.x / cm
            )

            dy = (
                state["creation_y_cm"]
                -
                vertex.y / cm
            )

            dz = (
                state["creation_z_cm"]
                -
                vertex.z / cm
            )

            position_score = (
                dx * dx
                +
                dy * dy
                +
                dz * dz
            )

            # ------------------------------------------------
            # Direction difference
            # ------------------------------------------------

            ddx = (
                state["direction_x"]
                -
                direction.x
            )

            ddy = (
                state["direction_y"]
                -
                direction.y
            )

            ddz = (
                state["direction_z"]
                -
                direction.z
            )

            direction_score = (
                ddx * ddx
                +
                ddy * ddy
                +
                ddz * ddz
            )

            # ------------------------------------------------
            # Energy difference
            # ------------------------------------------------

            energy_difference = abs(
                state["energy_eV"]
                -
                energy / eV
            )

            # ------------------------------------------------
            # Combined score
            #
            # The vertex and direction identify the photon.
            # Energy provides an additional constraint.
            # ------------------------------------------------

            score = (
                position_score
                +
                direction_score
                +
                (
                    energy_difference
                    *
                    energy_difference
                )
            )

            if score < best_score:

                best_score = score

                best_state = state

        # ----------------------------------------------------
        # Remove the state once assigned.
        #
        # This prevents one creation record from being used
        # for more than one optical track.
        # ----------------------------------------------------

        if best_state is not None:

            self.eventAction.pending_muon_states.remove(
                best_state
            )

        return best_state


    # ========================================================
    # New track
    # ========================================================

    def PreUserTrackingAction(self, track):

        event = (
            G4EventManager
            .GetEventManager()
            .GetConstCurrentEvent()
        )

        if event is None:

            return

        event_id = event.GetEventID()

        particle = (
            track.GetParticleDefinition()
        )

        # ====================================================
        # MUON TRACK
        # ====================================================

        if (
            particle ==
            G4MuonMinus.MuonMinusDefinition()
        ):

            # Only save the first muon.
            if self.eventAction.muon_truth is not None:

                return

            position = (
                track.GetVertexPosition()
            )

            direction = (
                track.GetVertexMomentumDirection()
            )

            kinetic_energy = (
                track.GetVertexKineticEnergy()
                / MeV
            )

            self.eventAction.muon_truth = {

                # Initial muon position
                "x":
                    position.x / cm,

                "y":
                    position.y / cm,

                "z":
                    position.z / cm,

                # Initial muon direction
                "dx":
                    direction.x,

                "dy":
                    direction.y,

                "dz":
                    direction.z,

                # Initial momentum is not needed by the
                # reconstruction, but retained.
                "px":
                    direction.x,

                "py":
                    direction.y,

                "pz":
                    direction.z,

                "energy":
                    kinetic_energy,

                # Water crossing information
                "water_in_x":
                    None,

                "water_in_y":
                    None,

                "water_in_z":
                    None,

                "water_out_x":
                    None,

                "water_out_y":
                    None,

                "water_out_z":
                    None,

                # Track length inside InnerWater
                "track_length":
                    0.0,

                # Internal state
                "_water_inside":
                    False,

                "_last_position":
                    position,

                "_last_volume":
                    None,

                "end_x":
                    position.x / cm,

                "end_y":
                    position.y / cm,

                "end_z":
                    position.z / cm,

                "end_energy":
                    kinetic_energy
            }

            return


        # ====================================================
        # OPTICAL PHOTON
        # ====================================================

        if (
            particle !=
            G4OpticalPhoton.OpticalPhotonDefinition()
        ):

            return

        # ----------------------------------------------------
        # Only Cherenkov photons
        # ----------------------------------------------------

        creator = (
            track.GetCreatorProcess()
        )

        if creator is None:

            return

        if (
            creator.GetProcessName()
            !=
            "Cerenkov"
        ):

            return

        # ----------------------------------------------------
        # Actual Geant4 track ID
        # ----------------------------------------------------

        track_id = (
            track.GetTrackID()
        )

        parent_id = (
            track.GetParentID()
        )

        # ----------------------------------------------------
        # Assign a unique photon ID
        # ----------------------------------------------------

        photon_id = (
            self.eventAction.photon_counter
        )

        self.eventAction.photon_counter += 1

        # ----------------------------------------------------
        # ACTUAL photon vertex
        # ----------------------------------------------------

        vertex = (
            track.GetVertexPosition()
        )

        direction = (
            track.GetVertexMomentumDirection()
        )

        energy = (
            track.GetVertexKineticEnergy()
        )

        # ----------------------------------------------------
        # Recover local muon state belonging to this photon
        # ----------------------------------------------------

        muon_state = self.FindMuonState(
            vertex,
            direction,
            energy
        )

        if muon_state is not None:

            muon_direction_x = (
                muon_state["muon_direction_x"]
            )

            muon_direction_y = (
                muon_state["muon_direction_y"]
            )

            muon_direction_z = (
                muon_state["muon_direction_z"]
            )

            muon_x_cm = (
                muon_state["muon_x_cm"]
            )

            muon_y_cm = (
                muon_state["muon_y_cm"]
            )

            muon_z_cm = (
                muon_state["muon_z_cm"]
            )

        else:

            muon_direction_x = None

            muon_direction_y = None

            muon_direction_z = None

            muon_x_cm = None

            muon_y_cm = None

            muon_z_cm = None

        # ----------------------------------------------------
        # Store exact optical track truth
        # ----------------------------------------------------

        photon_truth = {

            "event":
                event_id,

            "photon_id":
                photon_id,

            "track_id":
                track_id,

            "parent_id":
                parent_id,

            "creator":
                creator.GetProcessName(),

            # Actual Geant4 optical-photon vertex
            "creation_x_cm":
                vertex.x / cm,

            "creation_y_cm":
                vertex.y / cm,

            "creation_z_cm":
                vertex.z / cm,

            # Actual Geant4 optical-photon initial direction
            "direction_x":
                direction.x,

            "direction_y":
                direction.y,

            "direction_z":
                direction.z,

            # Actual optical-photon energy
            "energy_eV":
                energy / eV,

            # Local muon state at photon creation
            "muon_direction_x":
                muon_direction_x,

            "muon_direction_y":
                muon_direction_y,

            "muon_direction_z":
                muon_direction_z,

            "muon_x_cm":
                muon_x_cm,

            "muon_y_cm":
                muon_y_cm,

            "muon_z_cm":
                muon_z_cm,

            # Filled later by PMT matching
            "hit_pmt_id":
                None,

            "hit_x_cm":
                None,

            "hit_y_cm":
                None,

            "hit_z_cm":
                None,

            "hit_time_ns":
                None
        }

        # ----------------------------------------------------
        # Save photon truth
        # ----------------------------------------------------

        self.eventAction.optical_truth.append(
            photon_truth
        )

        # ----------------------------------------------------
        # DIRECT mapping
        #
        # No position matching.
        # No direction matching.
        # No energy matching.
        #
        # The track ID belongs to this exact optical photon.
        # ----------------------------------------------------

        self.eventAction.track_to_photon[
            (
                event_id,
                track_id
            )
        ] = photon_id


# ============================================================
# Stepping Action
# ============================================================

class SteppingAction(G4UserSteppingAction):

    def __init__(self, eventAction):

        super().__init__()

        self.eventAction = eventAction


    # ========================================================
    # Step
    # ========================================================

    def UserSteppingAction(self, step):

        track = (
            step.GetTrack()
        )

        particle = (
            track.GetParticleDefinition()
        )

        # Only interested in the muon for event truth.
        if (
            particle !=
            G4MuonMinus.MuonMinusDefinition()
        ):

            return

        truth = (
            self.eventAction.muon_truth
        )

        if truth is None:

            return

        # ----------------------------------------------------
        # Step positions
        # ----------------------------------------------------

        pre_point = (
            step.GetPreStepPoint()
        )

        post_point = (
            step.GetPostStepPoint()
        )

        pre_position = (
            pre_point.GetPosition()
        )

        post_position = (
            post_point.GetPosition()
        )

        # ----------------------------------------------------
        # Volumes
        #
        # GetPhysicalVolume() is not available in the Python
        # binding, so use Touchable -> Volume.
        # ----------------------------------------------------

        pre_touchable = (
            pre_point.GetTouchable()
        )

        post_touchable = (
            post_point.GetTouchable()
        )

        pre_volume = None
        post_volume = None

        if pre_touchable is not None:

            pre_volume = (
                pre_touchable.GetVolume()
            )

        if post_touchable is not None:

            post_volume = (
                post_touchable.GetVolume()
            )

        pre_name = None
        post_name = None

        if pre_volume is not None:

            pre_name = (
                pre_volume.GetName()
            )

        if post_volume is not None:

            post_name = (
                post_volume.GetName()
            )

        # ====================================================
        # ENTER WATER
        # ====================================================

        if (
            not truth["_water_inside"]
            and
            post_name == "InnerWater"
        ):

            truth["water_in_x"] = (
                post_position.x / cm
            )

            truth["water_in_y"] = (
                post_position.y / cm
            )

            truth["water_in_z"] = (
                post_position.z / cm
            )

            truth["_water_inside"] = True

        # ====================================================
        # TRACK LENGTH INSIDE WATER
        # ====================================================

        if (
            pre_name == "InnerWater"
        ):

            step_length = (
                post_position
                -
                pre_position
            ).mag()

            truth["track_length"] += (
                step_length / cm
            )

        # ====================================================
        # EXIT WATER
        # ====================================================

        if (
            truth["_water_inside"]
            and
            pre_name == "InnerWater"
            and
            post_name != "InnerWater"
        ):

            truth["water_out_x"] = (
                post_position.x / cm
            )

            truth["water_out_y"] = (
                post_position.y / cm
            )

            truth["water_out_z"] = (
                post_position.z / cm
            )

            truth["_water_inside"] = False

        # ====================================================
        # CURRENT FINAL POSITION / ENERGY
        # ====================================================

        truth["end_x"] = (
            post_position.x / cm
        )

        truth["end_y"] = (
            post_position.y / cm
        )

        truth["end_z"] = (
            post_position.z / cm
        )

        truth["end_energy"] = (
            track.GetKineticEnergy()
            / MeV
        )

        # ====================================================
        # CHERENKOV PHOTONS CREATED IN THIS MUON STEP
        #
        # Store the local muon state here.
        #
        # The actual optical track is handled later by
        # TrackingAction. This information is only used to
        # populate the local-muon truth fields.
        # ====================================================

        secondaries = (
            step.GetSecondaryInCurrentStep()
        )

        if secondaries is None:

            return

        muon_direction = (
            track.GetMomentumDirection()
        )

        muon_position = (
            track.GetPosition()
        )

        for secondary in secondaries:

            secondary_particle = (
                secondary.GetParticleDefinition()
            )

            if (
                secondary_particle
                !=
                G4OpticalPhoton.OpticalPhotonDefinition()
            ):

                continue

            creator = (
                secondary.GetCreatorProcess()
            )

            if creator is None:

                continue

            if (
                creator.GetProcessName()
                !=
                "Cerenkov"
            ):

                continue

            photon_position = (
                secondary.GetPosition()
            )

            photon_direction = (
                secondary.GetMomentumDirection()
            )

            photon_energy = (
                secondary.GetTotalEnergy()
            )

            self.eventAction.pending_muon_states.append({

                "creation_x_cm":
                    photon_position.x / cm,

                "creation_y_cm":
                    photon_position.y / cm,

                "creation_z_cm":
                    photon_position.z / cm,

                "direction_x":
                    photon_direction.x,

                "direction_y":
                    photon_direction.y,

                "direction_z":
                    photon_direction.z,

                "energy_eV":
                    photon_energy / eV,

                "muon_direction_x":
                    muon_direction.x,

                "muon_direction_y":
                    muon_direction.y,

                "muon_direction_z":
                    muon_direction.z,

                "muon_x_cm":
                    muon_position.x / cm,

                "muon_y_cm":
                    muon_position.y / cm,

                "muon_z_cm":
                    muon_position.z / cm
            })
