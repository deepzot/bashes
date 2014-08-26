#include "bashes/EstimatorEngine.h"

#include "boost/python.hpp"

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

BOOST_PYTHON_MODULE(_bashes) {
    bashes::pyExportEstimatorEngine();
}