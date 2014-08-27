#include "boost/python.hpp"

// See http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// See http://docs.scipy.org/doc/numpy/reference/c-api.array.html#miscellaneous
#define PY_ARRAY_UNIQUE_SYMBOL bashes_ARRAY_API
#include "numpy/arrayobject.h"

namespace bp = boost::python;

namespace bashes {
	void pyExportEstimatorEngine();
	void pyExportEstimatorHelpers();
}

BOOST_PYTHON_MODULE(_bashes) {
	// initialize the Numpy C API
    import_array();

    bashes::pyExportEstimatorEngine();
    bashes::pyExportEstimatorHelpers();
}