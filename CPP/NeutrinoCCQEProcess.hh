#ifndef NeutrinoCCQEProcess_h
#define NeutrinoCCQEProcess_h

//#include <G4VProcess.hh>
#include <G4VDiscreteProcess.hh>
#include <G4ParticleChange.hh>

class NeutrinoCCQEProcess : public G4VDiscreteProcess
{
public:

    NeutrinoCCQEProcess();

    ~NeutrinoCCQEProcess() override;

    G4bool IsApplicable(
        const G4ParticleDefinition&
    ) override;


    G4double PostStepGetPhysicalInteractionLength(
        const G4Track&,
        G4double,
        G4ForceCondition*
    ) override;
    G4double GetMeanFreePath(
        const G4Track&,
        G4double,
        G4ForceCondition*
    ) override;
    G4VParticleChange* PostStepDoIt(
        const G4Track&,
        const G4Step&
    ) override;

private:

    G4ParticleChange* pParticleChange;
};

#endif
