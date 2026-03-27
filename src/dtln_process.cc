#include "dtln.h"
#include "sfft.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

const int BLOCK_LEN = 512;
const int BLOCK_SHIFT = 128;
const int STATE_SIZE = 512;

void processDTLN(const std::string& model1Path,
                 const std::string& model2Path,
                 const std::vector<float>& inputAudio,
                 std::vector<float>& outputAudio,
                 bool showStats) {
    std::unique_ptr<tflite::FlatBufferModel> model1 =
        tflite::FlatBufferModel::BuildFromFile(model1Path.c_str());
    std::unique_ptr<tflite::FlatBufferModel> model2 =
        tflite::FlatBufferModel::BuildFromFile(model2Path.c_str());
    if (!model1 || !model2) {
        throw std::runtime_error("Failed to load models");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter1;
    std::unique_ptr<tflite::Interpreter> interpreter2;
    tflite::InterpreterBuilder(*model1, resolver)(&interpreter1);
    tflite::InterpreterBuilder(*model2, resolver)(&interpreter2);

    if (!interpreter1 || !interpreter2 ||
        interpreter1->AllocateTensors() != kTfLiteOk ||
        interpreter2->AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("Failed to create interpreters");
    }

    float* m1_in_spec = interpreter1->typed_input_tensor<float>(0);
    float* m1_in_state = interpreter1->typed_input_tensor<float>(1);
    float* m2_in_time = interpreter2->typed_input_tensor<float>(0);
    float* m2_in_state = interpreter2->typed_input_tensor<float>(1);
    std::memset(m1_in_state, 0, STATE_SIZE * sizeof(float));
    std::memset(m2_in_state, 0, STATE_SIZE * sizeof(float));

    int pad = BLOCK_LEN - BLOCK_SHIFT;
    std::vector<float> paddedAudio(inputAudio.size() + 2 * pad, 0.0f);
    std::copy(inputAudio.begin(), inputAudio.end(), paddedAudio.begin() + pad);

    std::vector<float> processedAudio(paddedAudio.size(), 0.0f);
    std::vector<std::complex<float>> fftBuffer(BLOCK_LEN);
    double totalTime = 0.0;
    int blockCount = 0;

    for (size_t idx = 0; idx <= paddedAudio.size() - BLOCK_LEN; idx += BLOCK_SHIFT) {
        blockCount++;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < BLOCK_LEN; ++i) {
            fftBuffer[i] = std::complex<float>(paddedAudio[idx + i], 0.0f);
        }
        SFFT::fft_512(fftBuffer);

        for (int i = 0; i <= BLOCK_LEN / 2; ++i) {
            m1_in_spec[i] = std::abs(fftBuffer[i]);
        }

        if (interpreter1->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Model1 inference failed");
        }

        const float* mask = interpreter1->typed_output_tensor<float>(0);
        const float* state1_out = interpreter1->typed_output_tensor<float>(1);
        std::memcpy(m1_in_state, state1_out, STATE_SIZE * sizeof(float));

        for (int i = 0; i <= BLOCK_LEN / 2; ++i) {
            fftBuffer[i] *= mask[i];
            if (i > 0 && i < BLOCK_LEN / 2) {
                fftBuffer[BLOCK_LEN - i] *= mask[i];
            }
        }

        SFFT::ifft_512(fftBuffer);
        for (int i = 0; i < BLOCK_LEN; ++i) {
            m2_in_time[i] = fftBuffer[i].real();
        }

        if (interpreter2->Invoke() != kTfLiteOk) {
            throw std::runtime_error("Model2 inference failed");
        }

        const float* enhanced = interpreter2->typed_output_tensor<float>(0);
        const float* state2_out = interpreter2->typed_output_tensor<float>(1);
        std::memcpy(m2_in_state, state2_out, STATE_SIZE * sizeof(float));

        for (int i = 0; i < BLOCK_LEN; ++i) {
            processedAudio[idx + i] += enhanced[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration<double, std::milli>(end - start).count();
    }

    outputAudio.resize(inputAudio.size());
    std::copy(processedAudio.begin() + pad,
              processedAudio.begin() + pad + inputAudio.size(),
              outputAudio.begin());

    float maxVal = *std::max_element(outputAudio.begin(), outputAudio.end(),
                                     [](float a, float b) {
                                         return std::abs(a) < std::abs(b);
                                     });

    if (std::abs(maxVal) > 1.0f) {
        float scale = 1.0f / std::abs(maxVal);
        for (auto& val : outputAudio) {
            val *= scale;
        }
    }

    if (showStats) {
        std::cout << "Total inference time: " << totalTime << " ms" << std::endl;
        std::cout << "Processed blocks: " << blockCount << std::endl;
        std::cout << "Avg block time: " << totalTime / blockCount << " ms" << std::endl;
    }
}
