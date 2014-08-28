// Created 22-Aug-2014 by David Kirkby (University of California, Irvine) <dkirkby@uci.edu>

#ifndef ESTIMATOR_ENGINE
#define ESTIMATOR_ENGINE

namespace bashes {
    class EstimatorEngine {
    public:
        EstimatorEngine();
        virtual ~EstimatorEngine();
        double timesTwo(double value) const;
    }; // EstimatorEngine
} // bashes

#endif // ESTIMATOR_ENGINE
