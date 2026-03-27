#pragma once
#include <complex>
#include <vector>

namespace SFFT {

    using complexf = std::complex<float>;
    constexpr int FFT_SIZE = 512;

    complexf* get_buf();
    void init();
    void cleanup();

    void fft_512(complexf* data);
    void ifft_512(complexf* data);

    void fft_512(std::vector<complexf>& data);
    void ifft_512(std::vector<complexf>& data);

}  // namespace SFFT
