// Created 22-Aug-2014 by David Kirkby (University of California, Irvine) <dkirkby@uci.edu>

#ifndef ESTIMATOR
#define ESTIMATOR

namespace bashes {
    class Estimator {
    public:
        Estimator();
        virtual ~Estimator();
        double timesTwo(double value) const;
    }; // Estimator
} // bashes

#endif // ESTIMATOR
