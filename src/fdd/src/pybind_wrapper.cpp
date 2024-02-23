#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "fdd/particle_filter.hpp"

namespace py = pybind11;
using namespace FDD;

PYBIND11_MODULE(FDDParticleFilter, m) {
    py::class_<ParticleFilter>(m, "ParticleFilter")
        .def(py::init<int, int, double>())
        .def("Loop", &ParticleFilter::Loop)
        .def("SetInitState", &ParticleFilter::SetInitState)
        .def("GetStateMatrix",&ParticleFilter::GetStateMatrix);
}