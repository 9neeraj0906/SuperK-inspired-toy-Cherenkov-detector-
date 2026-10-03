from geant4_pybind import *


class PhysicsList(NuBeam):

    def __init__(self):
        super().__init__()

        self.RegisterPhysics(
            G4OpticalPhysics()
        )

        self.RegisterPhysics(
            NeutrinoCCQEPhysics()
        )
