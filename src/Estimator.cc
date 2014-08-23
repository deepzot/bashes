// Created 22-Aug-2014 by David Kirkby (University of California, Irvine) <dkirkby@uci.edu>

#include "src/Estimator.h"

#include <iostream>

namespace bashes {

	Estimator::Estimator() {
		std::cout << "Estimator ctor" << std::endl;
	}

	Estimator::~Estimator() {
		std::cout << "Estimator dtor" << std::endl;
	}

	double Estimator::timesTwo(double value) const {
		return 2*value;
	}

} // bashes
