#ifndef WAVEBLOCKS_CONTINUOUS_SQRT
#define WAVEBLOCKS_CONTINUOUS_SQRT

#include <complex>
#include <stdexcept>

#include "math_util.hpp"

namespace waveblocks {

template<class T>
class ContinuousSqrt
{
private:
    /**
     * stored square root
     */
    std::complex<T> sqrt_;
    
    /**
     * argument (angle) of stored square root
     * domain = [-pi;pi]
     */
    T state_;
    
public:
    ContinuousSqrt()
        : sqrt_()
        , state_()
    { }
    
    ContinuousSqrt(std::complex<T> sqrt)
        : sqrt_(sqrt)
        , state_(std::arg(sqrt))
    { }
    
    ContinuousSqrt(const ContinuousSqrt &that) 
        : sqrt_(that.sqrt_)
        , state_(that.state_)
    { }
    
    ContinuousSqrt &operator=(const ContinuousSqrt &that)
    {
        sqrt_ = that.sqrt_;
        state_ = that.state_;
        return *this;
    }
    
    
    /**
     * Chooses the square root angle (aka argument) that continuates the reference angle the best.
     * Throws an exception if the deviation above an accepted value (by default >45°) as this
     * strongly indicates a problem in higher level code (for example a too large timestep).
     * \param[in] ref angle of reference root. domain = [-pi;pi]
     * \param[in] arg angle of  root. domain = [-pi;pi]
     * \return angle of continuating root. domain = [-pi;pi]
     */
    static T continuate(T ref, T arg)
    {
        const T PI = pi<T>();
        const T RANGE = 0.25*PI; // 0.5*pi allows all inputs
        
        //determine, how long one needs to 
        //rotate the reference angle counter-clock-wise to hit the angle of the 1st root
        T rot = arg - ref; // domain = [-2*pi;2*pi]
        
        // force rotation into domain [-pi;pi]
        if (rot >= PI)
            rot -= 2.0*PI;
        else if (rot < -PI)
            rot += 2.0*PI;
        
        if (rot > -RANGE && rot < RANGE) {
            return arg;
        }
        else if (rot > PI-RANGE || rot < -PI+RANGE) {
            if (arg > 0.0)
                return arg - PI;
            else
                return arg + PI;
        }
        else {
            throw std::runtime_error("continuous_sqrt: too large step");
        }
    }
    
    /**
     * update stored square root
     */
    std::complex<T> operator()(std::complex<T> input)
    {
        const T range = 0.25*pi<T>(); // 0.5*pi allows all inputs
        
        state_ = continuate(state_, 0.5*std::arg(input) );
        
        sqrt_ = std::polar(std::sqrt(std::abs(input)), state_);
        
        return sqrt_;
    }
    
    /**
     * retrieve stored square root
     */
    std::complex<T> operator()() const
    {
        return sqrt_;
    }
};

}

#endif