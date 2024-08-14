#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quadrotors_sim/vec_env.hpp"

namespace py = pybind11;
using namespace quadrotors;

PYBIND11_MODULE(MyQuadrotorEnv, m) {
    py::class_<VecEnv>(m, "VecEnv")
        .def(py::init<>())
        .def("step", &VecEnv::step)
        .def("get_num_envs", &VecEnv::get_num_envs)
        .def("get_obs_dim", &VecEnv::get_obs_dim)
        .def("get_action_dim", &VecEnv::get_action_dim)
        .def("get_state_dim", &VecEnv::get_state_dim)
        .def("reset", &VecEnv::reset)
        .def("set_k", &VecEnv::set_k)
        .def("get_k", &VecEnv::get_k)
        .def("set_state", &VecEnv::set_state)
        .def("get_state", &VecEnv::get_state)
        .def("getExtraInfoNames", &VecEnv::getExtraInfoNames)
        .def("print_quad_param", &VecEnv::print_quad_param);
}