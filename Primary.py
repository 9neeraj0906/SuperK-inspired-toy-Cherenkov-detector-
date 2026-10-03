from geant4_pybind import *


class PrimaryGeneratorAction(G4VUserPrimaryGeneratorAction):

    def __init__(self):
        super().__init__()

        self.particleGun = G4ParticleGun(1)

        self.neutrino = G4NeutrinoMu.NeutrinoMuDefinition()

        self.position = G4ThreeVector(
            0,
            0,
            5 * cm
        )

        self.direction = G4ThreeVector(
            0,
            0,
            -1
        )

    def GeneratePrimaries(self, event):

        self.particleGun.SetParticleDefinition(
            self.neutrino
        )

        self.particleGun.SetParticleEnergy(
            500 * MeV
        )

        self.particleGun.SetParticlePosition(
            self.position
        )

        self.particleGun.SetParticleMomentumDirection(
            self.direction
        )

        self.particleGun.GeneratePrimaryVertex(event)
