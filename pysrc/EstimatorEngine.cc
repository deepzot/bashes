// g++ -I/Users/daniel/source -I/Users/daniel/anaconda/include/python2.7 -fPIC -c Estimator.cc
// g++ -dynamiclib -dynamic Estimator.o -lboost_python -L/Users/daniel/anaconda/lib -L/Users/daniel/source/bashes/build -lpython2.7 -lbashes -o _bashes.so
// sudo install_name_tool -change build/libbashes.dylib /Users/daniel/source/bashes/build/libbashes.dylib _bashes.so

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