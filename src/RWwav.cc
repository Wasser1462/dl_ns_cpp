#include "RWwav.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>

bool readWav(const std::string& filename, WavHeader& header, std::vector<float>& audioData) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) return false;

    fin.read(reinterpret_cast<char*>(&header), sizeof(WavHeader));
    
    uint32_t dataStartPos = sizeof(WavHeader);
    uint32_t dataChunkSize = header.dataLength;

    if (strncmp(header.dataTag, "data", 4) != 0) {
        fin.seekg(12, std::ios::beg);
        while (true) {
            char chunkID[4];
            uint32_t chunkSize;
            fin.read(chunkID, 4);
            if (fin.eof()) break;
            fin.read(reinterpret_cast<char*>(&chunkSize), 4);
            
            if (strncmp(chunkID, "data", 4) == 0) {
                dataChunkSize = chunkSize;
                std::memcpy(header.dataTag, "data", 4);
                header.dataLength = chunkSize;
                dataStartPos = static_cast<uint32_t>(fin.tellg());
                break;
            }
            fin.seekg(chunkSize, std::ios::cur);
        }
    }
    
    if (strncmp(header.riffTag, "RIFF", 4) != 0 || 
        strncmp(header.waveTag, "WAVE", 4) != 0) {
        return false;
    }
    
    if (header.audioFormat != 1 || header.bitsPerSample != 16) {
        return false;
    }
    
    fin.seekg(dataStartPos, std::ios::beg);
    
    uint32_t bytesPerFrame = (header.bitsPerSample / 8) * header.numChannels;
    uint32_t numFrames = dataChunkSize / bytesPerFrame;
    audioData.resize(numFrames);
    
    for (uint32_t i = 0; i < numFrames; ++i) {
        int16_t sampleL = 0;
        fin.read(reinterpret_cast<char*>(&sampleL), sizeof(int16_t));
        
        if (header.numChannels == 2) {
            int16_t sampleR = 0;
            fin.read(reinterpret_cast<char*>(&sampleR), sizeof(int16_t));
            int32_t avg = (static_cast<int32_t>(sampleL) + static_cast<int32_t>(sampleR)) / 2;
            audioData[i] = avg / 32768.0f;
        } else {
            audioData[i] = sampleL / 32768.0f;
        }
    }
    
    return true;
}

bool writeWav(const std::string& filename, const WavHeader& header, const std::vector<float>& audioData) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) return false;

    WavHeader outHeader = header;
    std::memcpy(outHeader.riffTag, "RIFF", 4);
    std::memcpy(outHeader.waveTag, "WAVE", 4);
    std::memcpy(outHeader.fmtTag, "fmt ", 4);
    std::memcpy(outHeader.dataTag, "data", 4);
    
    outHeader.numChannels = 1;
    outHeader.bitsPerSample = 16;
    outHeader.blockAlign = outHeader.numChannels * (outHeader.bitsPerSample / 8);
    outHeader.byteRate = outHeader.sampleRate * outHeader.blockAlign;
    outHeader.dataLength = static_cast<uint32_t>(audioData.size()) * sizeof(int16_t);
    outHeader.riffLength = 36 + outHeader.dataLength;

    fout.write(reinterpret_cast<char*>(&outHeader), sizeof(WavHeader));

    for (float v : audioData) {
        float clamped = std::max(-1.0f, std::min(1.0f, v));
        int16_t sample = static_cast<int16_t>(std::round(clamped * 32767.0f));
        fout.write(reinterpret_cast<char*>(&sample), sizeof(int16_t));
    }
    
    return true;
}