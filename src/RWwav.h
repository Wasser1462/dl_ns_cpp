#ifndef RWWAV_H
#define RWWAV_H

#include <string>
#include <vector>
#include <cstdint>

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

bool readWav(const std::string& filename, WavHeader& header, std::vector<float>& audioData);
bool writeWav(const std::string& filename, const WavHeader& header, const std::vector<float>& audioData);

#endif