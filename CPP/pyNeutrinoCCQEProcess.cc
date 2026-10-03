#include <pybind11/pybind11.h>

#include <G4ParticleDefinition.hh>
#include <G4ProcessManager.hh>
#include <G4NeutrinoMu.hh>

#include "NeutrinoCCQEProcess.hh"

namespace py = pybind11;


void RegisterNeutrinoCCQE(
    G4ParticleDefinition* neutrino
)
{
    G4ProcessManager* processManager =
        neutrino->GetProcessManager();

    processManager->AddDiscreteProcess(
        new NeutrinoCCQEProcess()
    );
}


void register_NeutrinoCCQE(py::module &m)
{
    m.def(
        "RegisterNeutrinoCCQE",
        &RegisterNeutrinoCCQE
    );
}
