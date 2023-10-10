#include <sjb/utils/UtilityFunctions.hpp>
#include <fstream>

void UtilityFunctions::printTuplesToFile(Tuple *tuples, uint64_t size) {
    // open a target file
    std::ofstream outputFile;
    outputFile.open("input.csv", std::ios_base::app);

    // loop over the array
    for (uint64_t i = 0; i < size; i++) {
        // append the file
        outputFile << tuples[i].key << "," << tuples[i].val << "," << tuples[i].ts << "\n";
    }
}
