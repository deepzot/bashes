// Created 22-Aug-2014 by David Kirkby (University of California, Irvine) <dkirkby@uci.edu>

#include "bashes/EstimatorEngine.h"

#include <iostream>

namespace bashes {

	EstimatorEngine::EstimatorEngine() {
		std::cout << "EstimatorEngine ctor" << std::endl;
	}

	EstimatorEngine::~EstimatorEngine() {
		std::cout << "EstimatorEngine dtor" << std::endl;
	}

	double EstimatorEngine::timesTwo(double value) const {
		return 2*value;
	}

} // bashes
