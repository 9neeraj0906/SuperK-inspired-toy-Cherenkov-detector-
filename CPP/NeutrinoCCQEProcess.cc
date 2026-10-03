#include "NeutrinoCCQEProcess.hh"

#include <G4DynamicParticle.hh>
#include <G4Track.hh>
#include <G4Step.hh>

#include <G4NeutrinoMu.hh>
#include <G4MuonMinus.hh>
#include <G4Proton.hh>

#include <G4ThreeVector.hh>
#include <G4SystemOfUnits.hh>


NeutrinoCCQEProcess::NeutrinoCCQEProcess()
    : G4VDiscreteProcess("NeutrinoCCQE")
{
    pParticleChange = new G4ParticleChange();
}


NeutrinoCCQEProcess::~NeutrinoCCQEProcess()
{
    delete pParticleChange;
}


G4bool NeutrinoCCQEProcess::IsApplicable(
    const G4ParticleDefinition& particle
)
{
    return (
        &particle ==
        G4NeutrinoMu::NeutrinoMuDefinition()
    );
}
G4double NeutrinoCCQEProcess::GetMeanFreePath(
    const G4Track&,
    G4double,
    G4ForceCondition* condition
)
{
    *condition = Forced;
    return 0.0;
}
G4double NeutrinoCCQEProcess::PostStepGetPhysicalInteractionLength(
    const G4Track&,
    G4double,
    G4ForceCondition* condition
)
{
    *condition = Forced;

    return 0.0;
}


G4VParticleChange* NeutrinoCCQEProcess::PostStepDoIt(
    const G4Track& track,
    const G4Step&
)
{
    pParticleChange->Initialize(track);

    G4ThreeVector direction =
        track.GetMomentumDirection();

    G4ThreeVector position =
        track.GetPosition();

    G4double time =
        track.GetGlobalTime();

    G4DynamicParticle* muon =
        new G4DynamicParticle(
            G4MuonMinus::MuonMinusDefinition(),
            direction,
            400.0 * MeV
        );

    G4DynamicParticle* proton =
        new G4DynamicParticle(
            G4Proton::ProtonDefinition(),
            direction,
            100.0 * MeV
        );

    pParticleChange->AddSecondary(
        muon,
        position,
        time
    );

    pParticleChange->AddSecondary(
        proton,
        position,
        time
    );

    G4cout << "CCQE: created mu- + proton" << G4endl;
    pParticleChange->ProposeTrackStatus(
        fStopAndKill
    );

    return pParticleChange;
}
