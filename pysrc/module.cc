#include "boost/python.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace bashes {
	void pyExportEstimatorEngine();
	void pyExportEstimatorHelpers();
}

BOOST_PYTHON_MODULE(_bashes) {
	// initialize the Numpy C API
    import_array();
    bp::numeric::array::set_module_and_type("numpy", "ndarray");

    bashes::pyExportEstimatorEngine();
    bashes::pyExportEstimatorHelpers();
}