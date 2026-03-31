#ifndef DTLN_PROCESS_H
#define DTLN_PROCESS_H

#include <string>
#include <vector>

void processDTLN(const std::string& model1Path,
                 const std::string& model2Path,
                 const std::vector<float>& inputAudio,
                 std::vector<float>& outputAudio,
                 bool showStats = false);

#endif
