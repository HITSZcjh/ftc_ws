#pragma once
#include <eigen3/Eigen/Eigen>
#include <random>

namespace quadrotors
{
    // --- Setup some Eigen quick shorthands ---

    // Define the scalar type used.
    using Scalar = float; // numpy float32
    static constexpr int Dynamic = Eigen::Dynamic;

    // Using shorthand for `Matrix<rows, cols>` with scalar type.
    template <int rows = Dynamic, int cols = Dynamic>
    using Matrix = Eigen::Matrix<Scalar, rows, cols>; // lets you do Matrix<3,3> or Matrix<3,1> etc;

    template <int rows = Dynamic>
    using Vector = Matrix<rows, 1>; // lets you do Vector<3> or Vector<4> etc;

    template <typename Derived, int rows = Dynamic>
    using VectorRef = Eigen::Ref<Eigen::Matrix<Derived, rows, 1>>;

    template <typename Derived, int rows = Dynamic, int cols = Dynamic>
    using TensorsRef = Eigen::Ref<Eigen::Matrix<Derived, rows, cols, 1>>;

    // Helper functions

    template <typename T>
    T clamp(const T &value, const T &min, const T &max)
    {
        if (value < min)
        {
            return min;
        }
        else if (value > max)
        {
            return max;
        }
        else
        {
            return value;
        }
    }

    // template <typename Derived>
    // Derived clamp(const Eigen::MatrixBase<Derived> &input, typename Derived::Scalar min, typename Derived::Scalar max)
    // {
    //     return input.cwiseMax(min).cwiseMin(max);
    // }

    inline Matrix<3, 3> R(Vector<4> q) // q:w,x,y,z
    {
        return (Matrix<3, 3>() << 1 - 2 * (q[2] * q[2] + q[3] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[1] * q[2] + q[0] * q[3]), 1 - 2 * (q[1] * q[1] + q[3] * q[3]), 2 * (q[2] * q[3] - q[0] * q[1]),
                2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
            .finished();
    }

    class LPF_t
    {
    public:
        Scalar ts;
        Scalar tau; // time constant
        Vector<> last_input;
        Vector<> last_output;
        LPF_t(Scalar tau, Scalar ts) : tau(tau), ts(ts) {};
        void calc(VectorRef<Scalar> input,
                  VectorRef<Scalar> output)
        {
            output = (ts * input + tau * last_output) / (tau + ts);
            last_output = output;
        }
        void calc_derivative(VectorRef<Scalar> input,
                             VectorRef<Scalar> output)
        {
            output = (input - last_input + tau * last_output) / (tau + ts);
            last_input = input;
            last_output = output;
        }
    };

    // template <typename T>
    // inline T sign_single(T x)
    // {
    //     return (T(0) < x) - (x < T(0));
    // }
    // template <typename Derived>
    // Derived sign(const Derived &m)
    // {
    //     return m.unaryExpr(&sign_single<typename Derived::Scalar>);
    // }

    enum IDX : int
    {
        // position
        P = 0,
        PX = 0,
        PY = 1,
        PZ = 2,
        NP = 3,
        // linear velocity
        V = 3,
        VX = 3,
        VY = 4,
        VZ = 5,
        NV = 3,
        // quaternion
        Q = 6,
        QW = 6,
        QX = 7,
        QY = 8,
        QZ = 9,
        NQ = 4,
        // body rate
        W = 10,
        WX = 10,
        WY = 11,
        WZ = 12,
        NW = 3,
        // thrusts_real
        THRUSTS_REAL = 13,
        NTHRUSTS = 4,
        // SIZE
        NX = 17,
        NU = NTHRUSTS,
        NOBS = 23
    };

}