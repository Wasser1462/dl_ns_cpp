#ifndef SFFT_H
#define SFFT_H

#include <vector>
#include <complex>

namespace SFFT {
    const int FFT_SIZE = 512;
    extern int bitrev512[FFT_SIZE];

    void init_bitrev();     
    void fft_512(std::vector<std::complex<float>>& data); 
}

#endif