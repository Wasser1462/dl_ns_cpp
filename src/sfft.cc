#include "sfft.h"
#include <algorithm>

namespace SFFT {
    int bitrev512[FFT_SIZE]; 
    void init_bitrev() {
        int n = FFT_SIZE;
        int log2n = 0;
        while ((1 << log2n) < n) log2n++;
        for (int i = 0; i < n; ++i) {
            int x = i, rev = 0;
            for (int j = 0; j < log2n; ++j) {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            bitrev512[i] = rev;
        }
    }

    void fft_512(std::vector<std::complex<float>>& data) {
        int N = FFT_SIZE;
        for (int i = 0; i < N; ++i) {
            int j = bitrev512[i];
            if (j > i) std::swap(data[i], data[j]);
        }
        
        const float PI = 3.14159265358979323846f;
        for (int len = 2; len <= N; len <<= 1) {
            float angle = -2 * PI / len;
            std::complex<float> wlen(cos(angle), sin(angle));
            for (int i = 0; i < N; i += len) {
                std::complex<float> w(1.0f, 0.0f);
                int half = len >> 1;
                for (int j = 0; j < half; ++j) {
                    auto u = data[i + j];
                    auto v = data[i + j + half] * w;
                    data[i + j] = u + v;
                    data[i + j + half] = u - v;
                    w *= wlen;
                }
            }
        }
    }
}