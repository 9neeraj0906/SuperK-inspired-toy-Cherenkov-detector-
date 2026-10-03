#include <pybind11/pybind11.h>

#include "NeutrinoCCQEPhysics.hh"

namespace py = pybind11;


void register_NeutrinoCCQEPhysics(py::module &m)
{
    py::class_<
        NeutrinoCCQEPhysics,
        G4VPhysicsConstructor
    >(
        m,
        "NeutrinoCCQEPhysics"
    )
    .def(
        py::init<>()
    );
}
