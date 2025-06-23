#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <complex>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;

const int block_len = 512;
const int block_shift = 128;
const int state_size = 512; 

static int bitrev512[block_len];

void init_bitrev() {
    int n = block_len;
    int log2n = 0;
    while ((1 << log2n) < n) log2n++;
    for (int i = 0; i < n; ++i) {
        int x = i;
        int rev = 0;
        for (int j = 0; j < log2n; ++j) {
            rev <<= 1;
            rev |= (x & 1);
            x >>= 1;
        }
        bitrev512[i] = rev;
    }
}

void fft_512(vector<complex<float>>& data) {
    int N = block_len;
    for (int i = 0; i < N; ++i) {
        int j = bitrev512[i];
        if (j > i) {
            swap(data[i], data[j]);
        }
    }
    const float PI = 3.14159265358979323846f;
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -2 * PI / len;
        complex<float> wlen(cos(angle), sin(angle));
        for (int i = 0; i < N; i += len) {
            complex<float> w(1.0f, 0.0f);
            int half = len >> 1;
            for (int j = 0; j < half; ++j) {
                complex<float> u = data[i + j];
                complex<float> v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }
}

struct WavHeader {
    char riffTag[4];
    uint32_t riffLength;
    char waveTag[4];
    char fmtTag[4];
    uint32_t fmtLength;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char dataTag[4];
    uint32_t dataLength;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <model1.tflite> <model2.tflite> <input.wav> [output.wav]" << endl;
        return 1;
    }
    string model1_path = argv[1];
    string model2_path = argv[2];
    string in_wav_path = argv[3];
    string out_wav_path;
    if (argc >= 5) {
        out_wav_path = argv[4];
    } else {
        out_wav_path = "enhanced_output.wav";
    }

    ifstream fin(in_wav_path, ios::binary);
    if (!fin) {
        cerr << "Error: Unable to open input file " << in_wav_path << endl;
        return 1;
    }
    WavHeader wavHeader;
    fin.read(reinterpret_cast<char*>(&wavHeader), sizeof(WavHeader));
    // cout << "The WAV header reading is completed. Header size=" << sizeof(WavHeader) << endl;
    // cout << "Current location of the file: " << fin.tellg() << endl;
    // cout << "Data label: " << string(wavHeader.dataTag, 4) << endl;

    if (strncmp(wavHeader.dataTag, "data", 4) != 0) {
        cout << "The file may contain additional information. Try to search for the data block..." << endl;
        auto currentPos = fin.tellg();
        fin.seekg(12, fin.beg);
        while (true) {
            char chunkID[4];
            uint32_t chunkSize;
            fin.read(chunkID, 4);
            if (fin.eof()) break;
            fin.read(reinterpret_cast<char*>(&chunkSize), 4);
            if (strncmp(chunkID, "data", 4) == 0) {
                wavHeader.dataLength = chunkSize;
                break;
            }
            fin.seekg(chunkSize, fin.cur);
        }
    }
    if (strncmp(wavHeader.riffTag, "RIFF", 4) != 0 || strncmp(wavHeader.waveTag, "WAVE", 4) != 0) {
        cerr << "Error: Invalid WAV file header" << endl;
        return 1;
    }
    if (wavHeader.audioFormat != 1) {
        cerr << "Error: Unsupported audio format: " << wavHeader.audioFormat << endl;
        return 1;
    }
    if (wavHeader.bitsPerSample != 16) {
        cerr << "Error: Only 16-bit PCM supported" << endl;
        return 1;
    }
    if (wavHeader.numChannels < 1 || wavHeader.numChannels > 2) {
        cerr << "Error: Only mono or stereo supported" << endl;
        return 1;
    }

    uint32_t totalDataBytes = wavHeader.dataLength;
    uint32_t numFrames = totalDataBytes / (wavHeader.bitsPerSample/8 * wavHeader.numChannels);
    vector<float> audio_data;
    audio_data.reserve(numFrames);
    for (uint32_t i = 0; i < numFrames; ++i) {
        int16_t sample = 0;
        int16_t sampleR = 0;
        fin.read(reinterpret_cast<char*>(&sample), sizeof(int16_t));
        if (wavHeader.numChannels == 2) {
            fin.read(reinterpret_cast<char*>(&sampleR), sizeof(int16_t));
            int32_t avg = (int32_t)sample + (int32_t)sampleR;
            avg /= 2;
            sample = static_cast<int16_t>(avg);
        }
        float fval = sample / 32768.0f;
        audio_data.push_back(fval);
    }
    fin.close();
    size_t len_audio = audio_data.size();
    if (len_audio == 0) {
        cerr << "Error: No audio samples read" << endl;
        return 1;
    }

    int pad = block_len - block_shift;
    vector<float> in_audio;
    in_audio.resize(len_audio + 2 * pad);
    for (int i = 0; i < pad; ++i) {
        in_audio[i] = 0.0f;
    }
    for (size_t i = 0; i < len_audio; ++i) {
        in_audio[pad + i] = audio_data[i];
    }
    for (int i = 0; i < pad; ++i) {
        in_audio[pad + len_audio + i] = 0.0f;
    }

    size_t total_len = in_audio.size();
    vector<float> out_audio;
    out_audio.resize(total_len);
    fill(out_audio.begin(), out_audio.end(), 0.0f);

    init_bitrev();
    vector<complex<float>> fft_data(block_len);

    unique_ptr<tflite::FlatBufferModel> model1 = tflite::FlatBufferModel::BuildFromFile(model1_path.c_str());
    unique_ptr<tflite::FlatBufferModel> model2 = tflite::FlatBufferModel::BuildFromFile(model2_path.c_str());
    if (!model1 || !model2) {
        cerr << "Error: Failed to load TFLite models" << endl;
        return 1;
    }
    tflite::ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<tflite::Interpreter> interpreter1;
    unique_ptr<tflite::Interpreter> interpreter2;

    tflite::InterpreterBuilder(*model1, resolver)(&interpreter1);
    tflite::InterpreterBuilder(*model2, resolver)(&interpreter2);
    
    if (!interpreter1 || !interpreter2) {
        cerr << "Error: Failed to create interpreters" << endl;
        return 1;
    }
    if (interpreter1->AllocateTensors() != kTfLiteOk || interpreter2->AllocateTensors() != kTfLiteOk) {
        cerr << "Error: Tensor allocation failed" << endl;
        return 1;
    }

    if (interpreter1->inputs().size() != 2) {
        cerr << "Error: Model1 expects 2 inputs" << endl;
        return 1;
    }
    TfLiteTensor* m1_in0 = interpreter1->input_tensor(0);
    int m1_in0_count = 1;
    for (int d = 0; d < m1_in0->dims->size; ++d) {
        m1_in0_count *= m1_in0->dims->data[d];
    }
    if (m1_in0_count != 257) {
        cerr << "Error: Model1 input0 size mismatch" << endl;
        return 1;
    }
    TfLiteTensor* m1_in1 = interpreter1->input_tensor(1);
    int m1_in1_count = 1;
    for (int d = 0; d < m1_in1->dims->size; ++d) {
        m1_in1_count *= m1_in1->dims->data[d];
    }
    if (m1_in1_count != state_size) {
        cerr << "Error: Model1 state size mismatch" << endl;
        return 1;
    }

    if (interpreter1->outputs().size() != 2) {
        cerr << "Error: Model1 expects 2 outputs" << endl;
        return 1;
    }

    TfLiteTensor* m1_out0 = interpreter1->output_tensor(0);
    int m1_out0_count = 1;
    for (int d = 0; d < m1_out0->dims->size; ++d) {
        m1_out0_count *= m1_out0->dims->data[d];
    }
    if (m1_out0_count != 257) {
        cerr << "Error: Model1 output0 size mismatch" << endl;
        return 1;
    }

    TfLiteTensor* m1_out1 = interpreter1->output_tensor(1);
    int m1_out1_count = 1;
    for (int d = 0; d < m1_out1->dims->size; ++d) {
        m1_out1_count *= m1_out1->dims->data[d];
    }
    if (m1_out1_count != state_size) {
        cerr << "Error: Model1 state output size mismatch" << endl;
        return 1;
    }

    if (interpreter2->inputs().size() != 2 || interpreter2->outputs().size() != 2) {
        cerr << "Error: Model2 expects 2 inputs and 2 outputs" << endl;
        return 1;
    }
    TfLiteTensor* m2_in0 = interpreter2->input_tensor(0);
    int m2_in0_count = 1;
    for (int d = 0; d < m2_in0->dims->size; ++d) {
        m2_in0_count *= m2_in0->dims->data[d];
    }
    if (m2_in0_count != block_len) {
        cerr << "Error: Model2 input0 size mismatch" << endl;
        return 1;
    }
    TfLiteTensor* m2_in1 = interpreter2->input_tensor(1);
    int m2_in1_count = 1;
    for (int d = 0; d < m2_in1->dims->size; ++d) {
        m2_in1_count *= m2_in1->dims->data[d];
    }
    if (m2_in1_count != state_size) {
        cerr << "Error: Model2 state size mismatch" << endl;
        return 1;
    }
    TfLiteTensor* m2_out0 = interpreter2->output_tensor(0);
    int m2_out0_count = 1;
    for (int d = 0; d < m2_out0->dims->size; ++d) {
        m2_out0_count *= m2_out0->dims->data[d];
    }
    if (m2_out0_count != block_len) {
        cerr << "Error: Model2 output0 size mismatch" << endl;
        return 1;
    }
    TfLiteTensor* m2_out1 = interpreter2->output_tensor(1);
    int m2_out1_count = 1;
    for (int d = 0; d < m2_out1->dims->size; ++d) {
        m2_out1_count *= m2_out1->dims->data[d];
    }
    if (m2_out1_count != state_size) {
        cerr << "Error: Model2 state output size mismatch" << endl;
        return 1;
    }

    float* m1_in_spec = interpreter1->typed_input_tensor<float>(0);
    float* m1_in_state = interpreter1->typed_input_tensor<float>(1);
    float* m2_in_time = interpreter2->typed_input_tensor<float>(0);
    float* m2_in_state = interpreter2->typed_input_tensor<float>(1);
    
    memset(m1_in_state, 0, state_size * sizeof(float));
    memset(m2_in_state, 0, state_size * sizeof(float));

    double total_inference_time = 0.0;
    int block_count = 0;

    for (size_t inIndex = 0; ; inIndex += block_shift) {
        if (inIndex > total_len - block_len) {
            break;
        }
        block_count++;

        for (int n = 0; n < block_len; ++n) {
            float sample = in_audio[inIndex + n];
            fft_data[n] = complex<float>(sample, 0.0f);
        }
        fft_512(fft_data);

        for (int k = 0; k <= block_len/2; ++k) {
            float real = fft_data[k].real();
            float imag = fft_data[k].imag();
            m1_in_spec[k] = sqrt(real*real + imag*imag);
        }

        auto start_inference = chrono::high_resolution_clock::now();
        if (interpreter1->Invoke() != kTfLiteOk) {
            cerr << "Error: Model1 inference failed" << endl;
            return 1;
        }
        auto end_inference = chrono::high_resolution_clock::now();
        double inference_time = chrono::duration<double, milli>(end_inference - start_inference).count();
        total_inference_time += inference_time;

        const float* mask_out = interpreter1->typed_output_tensor<float>(0);
        const float* state1_out = interpreter1->typed_output_tensor<float>(1);
        
        memcpy(m1_in_state, state1_out, state_size * sizeof(float));

        for (int k = 0; k <= block_len/2; ++k) {
            float mask_val = mask_out[k];
            fft_data[k] *= mask_val;
            if (k != 0 && k != block_len/2) {
                fft_data[block_len - k] *= mask_val;
            }
        }

        for (int i = 0; i < block_len; ++i) {
            fft_data[i] = conj(fft_data[i]);
        }
        fft_512(fft_data);
        for (int i = 0; i < block_len; ++i) {
            fft_data[i] = conj(fft_data[i]) * (1.0f / block_len);
        }

        for (int n = 0; n < block_len; ++n) {
            m2_in_time[n] = fft_data[n].real();
        }

        start_inference = chrono::high_resolution_clock::now();
        if (interpreter2->Invoke() != kTfLiteOk) {
            cerr << "Error: Model2 inference failed" << endl;
            return 1;
        }
        end_inference = chrono::high_resolution_clock::now();
        inference_time = chrono::duration<double, milli>(end_inference - start_inference).count();
        total_inference_time += inference_time;

        const float* enhanced_out = interpreter2->typed_output_tensor<float>(0);
        const float* state2_out = interpreter2->typed_output_tensor<float>(1);

        memcpy(m2_in_state, state2_out, state_size * sizeof(float));

        for (int n = 0; n < block_len; ++n) {
            out_audio[inIndex + n] += enhanced_out[n];
        }
    }

    vector<float> predicted;
    predicted.resize(len_audio);
    for (size_t i = 0; i < len_audio; ++i) {
        predicted[i] = out_audio[pad + i];
    }

    float maxVal = 0.0f;
    for (float v : predicted) {
        float av = fabs(v);
        if (av > maxVal) maxVal = av;
    }
    if (maxVal > 1.0f) {
        float scale = 1.0f / maxVal;
        for (float &v : predicted) {
            v *= scale;
        }
    }

    ofstream fout(out_wav_path, ios::binary);
    if (!fout) {
        cerr << "Error: Unable to create output file " << out_wav_path << endl;
        return 1;
    }

    WavHeader outHeader = wavHeader;
    memcpy(outHeader.riffTag, "RIFF", 4);
    memcpy(outHeader.waveTag, "WAVE", 4);
    memcpy(outHeader.fmtTag, "fmt ", 4);
    memcpy(outHeader.dataTag, "data", 4);  
    outHeader.numChannels = 1;
    outHeader.bitsPerSample = 16;
    outHeader.blockAlign = outHeader.numChannels * outHeader.bitsPerSample / 8;
    outHeader.sampleRate = wavHeader.sampleRate;
    outHeader.byteRate = outHeader.sampleRate * outHeader.blockAlign;
    outHeader.dataLength = predicted.size() * outHeader.numChannels * outHeader.bitsPerSample / 8;
    outHeader.riffLength = 36 + outHeader.dataLength;

    fout.write(reinterpret_cast<char*>(&outHeader), sizeof(WavHeader));

    for (float v : predicted) {
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t sample_out = static_cast<int16_t>(round(v * 32767.0f));
        fout.write(reinterpret_cast<char*>(&sample_out), sizeof(int16_t));
    }
    fout.close();

    cout << "Processing completed. Enhanced audio saved to " << out_wav_path << endl;
    cout << "Total model inference time: " << total_inference_time << " ms" << endl;
    cout << "Blocks processed: " << block_count << endl;
    cout << "Average inference time per block: " << total_inference_time/block_count << " ms" << endl;

    return 0;
}