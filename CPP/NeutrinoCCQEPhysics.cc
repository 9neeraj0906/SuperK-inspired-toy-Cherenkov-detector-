#include "NeutrinoCCQEPhysics.hh"
#include "NeutrinoCCQEProcess.hh"

#include <G4NeutrinoMu.hh>
#include <G4ProcessManager.hh>


NeutrinoCCQEPhysics::NeutrinoCCQEPhysics()
    : G4VPhysicsConstructor("NeutrinoCCQEPhysics")
{
}


NeutrinoCCQEPhysics::~NeutrinoCCQEPhysics()
{
}


void NeutrinoCCQEPhysics::ConstructParticle()
{
    // νμ is already defined by the physics list.
}


void NeutrinoCCQEPhysics::ConstructProcess()
{
    G4ParticleDefinition* neutrino =
        G4NeutrinoMu::NeutrinoMuDefinition();

    G4ProcessManager* processManager =
        neutrino->GetProcessManager();

    processManager->AddDiscreteProcess(
        new NeutrinoCCQEProcess()
    );
}
