#include "RWwav.h"
#include "dtln.h"

#include <cstring>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <model1.tflite> <model2.tflite> <input.wav> [output.wav]"
                  << std::endl;
        return 1;
    }

    WavHeader header;
    std::vector<float> inputAudio;
    if (!readWav(argv[3], header, inputAudio) || inputAudio.empty()) {
        std::cerr << "Error reading WAV file" << std::endl;
        return 1;
    }

    WavHeader outHeader;
    std::memcpy(outHeader.riffTag, "RIFF", 4);
    std::memcpy(outHeader.waveTag, "WAVE", 4);
    std::memcpy(outHeader.fmtTag, "fmt ", 4);
    std::memcpy(outHeader.dataTag, "data", 4);

    outHeader.fmtLength = 16;
    outHeader.audioFormat = 1;
    outHeader.numChannels = 1;
    outHeader.sampleRate = header.sampleRate;
    outHeader.bitsPerSample = 16;
    outHeader.blockAlign = outHeader.numChannels * (outHeader.bitsPerSample / 8);
    outHeader.byteRate = outHeader.sampleRate * outHeader.blockAlign;

    std::vector<float> outputAudio;
    try {
        processDTLN(argv[1], argv[2], inputAudio, outputAudio, true);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    outHeader.dataLength = outputAudio.size() * sizeof(int16_t);
    outHeader.riffLength = 4 + 24 + 8 + outHeader.dataLength;

    std::string outPath = (argc >= 5) ? argv[4] : "enhanced.wav";

    if (!writeWav(outPath, outHeader, outputAudio)) {
        std::cerr << "Error writing WAV file" << std::endl;
        return 1;
    }

    std::cout << "Enhanced audio saved to " << outPath << std::endl;
    return 0;
}
