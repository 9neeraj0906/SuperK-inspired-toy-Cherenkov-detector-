#ifndef NeutrinoCCQEPhysics_h
#define NeutrinoCCQEPhysics_h

#include <G4VPhysicsConstructor.hh>

class NeutrinoCCQEPhysics : public G4VPhysicsConstructor
{
public:

    NeutrinoCCQEPhysics();

    ~NeutrinoCCQEPhysics() override;

    void ConstructParticle() override;

    void ConstructProcess() override;
};

#endif
