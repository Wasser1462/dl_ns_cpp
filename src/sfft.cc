#include "sfft.h"
#include <fftw3.h>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace SFFT {

    static fftwf_plan fwd_plan = nullptr;
    static fftwf_plan bwd_plan = nullptr;
    static std::mutex plan_mtx;

    static thread_local complexf* buf = [] {
        auto p = reinterpret_cast<complexf*>(fftwf_alloc_complex(FFT_SIZE));
        if (!p) throw std::bad_alloc();
        return p;
    }();

    complexf* get_buf() {
        return buf;
    }

    inline bool aligned16(const void* p) {
        return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0;
    }

    void init() {
        std::lock_guard<std::mutex> lk(plan_mtx);
        if (fwd_plan) return;

        fftwf_complex* dummy = fftwf_alloc_complex(FFT_SIZE);
        if (!dummy) throw std::runtime_error("FFTW alloc failed");

        fwd_plan = fftwf_plan_dft_1d(
            FFT_SIZE, dummy, dummy,
            FFTW_FORWARD, FFTW_ESTIMATE);

        bwd_plan = fftwf_plan_dft_1d(
            FFT_SIZE, dummy, dummy,
            FFTW_BACKWARD, FFTW_ESTIMATE);

        if (!fwd_plan || !bwd_plan)
            throw std::runtime_error("FFTW plan creation failed");

        fftwf_free(dummy);
    }

    void cleanup() {
        std::lock_guard<std::mutex> lk(plan_mtx);
        if (fwd_plan) {
            fftwf_destroy_plan(fwd_plan);
            fwd_plan = nullptr;
        }
        if (bwd_plan) {
            fftwf_destroy_plan(bwd_plan);
            bwd_plan = nullptr;
        }
    }

    void fft_512(complexf* data) {
        init();
        fftwf_execute_dft(fwd_plan,
                          reinterpret_cast<fftwf_complex*>(data),
                          reinterpret_cast<fftwf_complex*>(data));
    }

    void ifft_512(complexf* data) {
        init();
        fftwf_execute_dft(bwd_plan,
                          reinterpret_cast<fftwf_complex*>(data),
                          reinterpret_cast<fftwf_complex*>(data));

        constexpr float scale = 1.0f / FFT_SIZE;
        for (int i = 0; i < FFT_SIZE; ++i) data[i] *= scale;
    }

    void fft_512(std::vector<complexf>& d) {
        if (d.size() != FFT_SIZE)
            throw std::runtime_error("fft_512: vector size != 512");

        if (aligned16(d.data())) {
            fft_512(d.data());
        } else {
            static thread_local complexf* scratch =
                reinterpret_cast<complexf*>(fftwf_alloc_complex(FFT_SIZE));
            std::memcpy(scratch, d.data(), FFT_SIZE * sizeof(complexf));
            fft_512(scratch);
            std::memcpy(d.data(), scratch, FFT_SIZE * sizeof(complexf));
        }
    }

    void ifft_512(std::vector<complexf>& d) {
        if (d.size() != FFT_SIZE)
            throw std::runtime_error("ifft_512: vector size != 512");

        if (aligned16(d.data())) {
            ifft_512(d.data());
        } else {
            static thread_local complexf* scratch =
                reinterpret_cast<complexf*>(fftwf_alloc_complex(FFT_SIZE));
            std::memcpy(scratch, d.data(), FFT_SIZE * sizeof(complexf));
            ifft_512(scratch);
            std::memcpy(d.data(), scratch, FFT_SIZE * sizeof(complexf));
        }
    }

}  // namespace SFFT
