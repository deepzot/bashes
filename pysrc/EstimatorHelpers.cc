#include "boost/python.hpp"
#include "boost/python/extract.hpp"
#include "boost/python/numeric.hpp"

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <iostream>

namespace bp = boost::python;
namespace bpn = boost::python::numeric;

namespace bashes {

namespace {

// Example using only boost::python::numeric::array
void printFirst(bpn::array data) {
    // Access a built-in type (an array)
    bpn::array a = data;
    // Need to <extract> array elements because their type is unknown
    std::cout << "First array item: " << bp::extract<int>(a[0]) << std::endl;
};

// Example using numpy c-api constructs
// see http://stackoverflow.com/questions/9128519/reading-many-values-from-numpy-c-api
bp::object timesTwo(bpn::array m){
    // access underlying numpy PyObject pointer of input array
    PyObject* m_obj = PyArray_FROM_OTF(m.ptr(), NPY_DOUBLE, NPY_IN_ARRAY);
    // to avoid memory leaks, let a Boost::Python object manage the array
    bp::object temp((bp::handle<>(m_obj)));
    // number of array dimensions
    int ndim = PyArray_NDIM(m_obj);
    std::cout << "bashes::timesTwo: ndim of input array: " << ndim << std::endl;
    // get direct access to the array data
    const double* data = static_cast<const double*>(PyArray_DATA(m_obj));
    // make the output array, and get access to its data
    PyObject* res = PyArray_SimpleNew(ndim, PyArray_DIMS(m_obj), NPY_DOUBLE);
    // access output array data
    double* res_data = static_cast<double*>(PyArray_DATA(res));
    // number of elements in array
    const unsigned size = PyArray_SIZE(m_obj); 
    std::cout << "bashes::timesTwo: size of input array: " << size << std::endl;
    // times by two
    for (unsigned i = 0; i < size; ++i) {
        res_data[i] = 2*data[i];
    }
    // go back to using Boost::Python constructs
    return bp::object((bp::handle<>(res)));
};

} // anonymous

void pyExportEstimatorHelpers() {
    bp::def("printFirst", &printFirst);
    bp::def("timesTwo", &timesTwo);
}

} // bashes