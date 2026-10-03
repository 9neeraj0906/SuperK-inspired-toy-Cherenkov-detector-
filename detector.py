from geant4_pybind import *
import math


MAX_STEP = 0.01 * mm


class PMTSensitiveDetector(G4VSensitiveDetector):

    def __init__(self):
        super().__init__("PMTSensitiveDetector")

        self.hits = []
        self.all_hits = []
        self.hit_count = 0
        self.printed_hits = 0

    def Initialize(self, hce):

        self.hits = []
        self.hit_count = 0
        self.printed_hits = 0

    def ProcessHits(self, step, history):

        track = step.GetTrack()
        particle = track.GetParticleDefinition()

        if particle != G4OpticalPhoton.OpticalPhotonDefinition():
            return False

        self.hit_count += 1

        touchable = step.GetPreStepPoint().GetTouchable()

        pmt_id = touchable.GetCopyNumber()

        position = step.GetPreStepPoint().GetPosition()

        hit_time = step.GetPreStepPoint().GetGlobalTime()

        energy = track.GetKineticEnergy()

        track_id = track.GetTrackID()

        parent_id = track.GetParentID()

        event = G4EventManager.GetEventManager().GetConstCurrentEvent()

        event_id = event.GetEventID()

        hit = {
            "event": event_id,
            "track_id": track_id,
            "parent_id": parent_id,
            "pmt_id": pmt_id,
            "x": position.x / cm,
            "y": position.y / cm,
            "z": position.z / cm,
            "time": hit_time / ns,
            "energy": energy / eV
        }

        self.hits.append(hit)

        self.all_hits.append(hit)

        if self.printed_hits < 5:

            print(
                "PMT HIT:",
                "Event =", event_id,
                "PMT ID =", pmt_id,
                "TrackID =", track_id,
                "ParentID =", parent_id,
                "Energy =", energy / eV,
                "eV"
            )

            print(
                "Position =",
                position.x / cm,
                position.y / cm,
                position.z / cm,
                "cm"
            )

            print(
                "Time =",
                hit_time / ns,
                "ns"
            )

            self.printed_hits += 1

        track.SetTrackStatus(fStopAndKill)

        return True


class CherenkovDetectorConstruction(G4VUserDetectorConstruction):

    def __init__(self):

        super().__init__()

        self.pmtSD = PMTSensitiveDetector()

    def Construct(self):

        nist = G4NistManager.Instance()

        # ==================================================
        # MATERIALS
        # ==================================================

        world_mat = nist.FindOrBuildMaterial("G4_AIR")

        water_mat = nist.FindOrBuildMaterial("G4_WATER")

        pmt_mat = nist.FindOrBuildMaterial("G4_GLASS_PLATE")

        # ==================================================
        # WORLD
        # ==================================================

        world_sizeXY = 40.0 * cm
        world_sizeZ = 40.0 * cm

        world_solid = G4Box(
            "World",
            world_sizeXY / 2,
            world_sizeXY / 2,
            world_sizeZ / 2
        )

        world_logical = G4LogicalVolume(
            world_solid,
            world_mat,
            "World"
        )

        world_physical = G4PVPlacement(
            None,
            G4ThreeVector(0, 0, 0),
            world_logical,
            "World",
            None,
            False,
            0,
            True
        )

        # ==================================================
        # OUTER WATER TANK
        # ==================================================

        tank_radius = 20.0 * cm
        tank_halfZ = 10.0 * cm

        tank_solid = G4Tubs(
            "WaterTank",
            0,
            tank_radius,
            tank_halfZ,
            0,
            360 * deg
        )

        tank_logical = G4LogicalVolume(
            tank_solid,
            water_mat,
            "WaterTank"
        )

        G4PVPlacement(
            None,
            G4ThreeVector(0, 0, 0),
            tank_logical,
            "WaterTank",
            world_logical,
            False,
            0,
            True
        )

        # ==================================================
        # INNER WATER VOLUME
        # ==================================================

        inner_radius = 10.0 * cm
        inner_halfZ = 5.0 * cm

        inner_solid = G4Tubs(
            "InnerWater",
            0,
            inner_radius,
            inner_halfZ,
            0,
            360 * deg
        )

        inner_logical = G4LogicalVolume(
            inner_solid,
            water_mat,
            "InnerWater"
        )

        G4PVPlacement(
            None,
            G4ThreeVector(0, 0, 0),
            inner_logical,
            "InnerWater",
            tank_logical,
            False,
            0,
            True
        )

        # ==================================================
        # WATER OPTICAL PROPERTIES
        # ==================================================

        photon_energy = G4doubleVector([
            1.77 * eV,
            2.07 * eV,
            2.48 * eV,
            2.76 * eV,
            3.10 * eV,
            3.54 * eV
        ])

        refractive_index = G4doubleVector([
            1.3318,
            1.3339,
            1.3350,
            1.3380,
            1.3435,
            1.3490
        ])

        absorption_length = G4doubleVector([
            1.61 * m,
            4.35 * m,
            41.32 * m,
            93.5 * m,
            142.9 * m,
            49.0 * m
        ])

        water_properties = G4MaterialPropertiesTable()

        water_properties.AddProperty(
            "RINDEX",
            photon_energy,
            refractive_index
        )

        water_properties.AddProperty(
            "ABSLENGTH",
            photon_energy,
            absorption_length
        )

        water_mat.SetMaterialPropertiesTable(
            water_properties
        )

        # ==================================================
        # PMT GLASS OPTICAL PROPERTIES
        # ==================================================

        pmt_properties = G4MaterialPropertiesTable()

        pmt_refractive_index = G4doubleVector([
            1.50,
            1.50,
            1.50,
            1.50,
            1.50,
            1.50
        ])

        pmt_properties.AddProperty(
            "RINDEX",
            photon_energy,
            pmt_refractive_index
        )

        pmt_mat.SetMaterialPropertiesTable(
            pmt_properties
        )

        # ==================================================
        # PMT GEOMETRY
        # ==================================================

        pmt_radius = 0.5 * cm

        pmt_solid = G4Tubs(
            "PMT",
            0,
            pmt_radius,
            0.05 * cm,
            0,
            360 * deg
        )

        pmt_logical = G4LogicalVolume(
            pmt_solid,
            pmt_mat,
            "PMT"
        )

        pmt_logical.SetSensitiveDetector(
            self.pmtSD
        )

        # ==================================================
        # 200 PMTs
        # ==================================================

        nPhi = 40
        nZ = 5

        pmt_radial_position = (
            inner_radius -
            pmt_radius
        )

        copy_number = 0

        for iz in range(nZ):

            if nZ == 1:

                z = 0

            else:

                z = (
                    -inner_halfZ +
                    pmt_radius +
                    iz * (
                        2 * (
                            inner_halfZ -
                            pmt_radius
                        )
                        / (nZ - 1)
                    )
                )

            for iphi in range(nPhi):

                phi = 2 * math.pi * iphi / nPhi

                x = pmt_radial_position * math.cos(phi)

                y = pmt_radial_position * math.sin(phi)

                G4PVPlacement(
                    None,
                    G4ThreeVector(x, y, z),
                    pmt_logical,
                    "PMT",
                    inner_logical,
                    False,
                    copy_number,
                    True
                )

                copy_number += 1

        return world_physical
