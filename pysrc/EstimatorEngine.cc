#include "boost/python.hpp"

#include "bashes/EstimatorEngine.h"

namespace bp = boost::python;

namespace bashes {

namespace {

struct PyEstimatorEngine {
    static void wrap() {
        bp::class_<EstimatorEngine> pyEstimatorEngine("EstimatorEngine", bp::init<>());
        pyEstimatorEngine
            .def("timesTwo", &EstimatorEngine::timesTwo, bp::arg("value"),
                "Returns the value times two.")
        ;
    }
};

} // anonymous

void pyExportEstimatorEngine() {
    PyEstimatorEngine::wrap();
}

} // bashes