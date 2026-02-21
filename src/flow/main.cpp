#include "pushrelabel.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    // Ensure we have enough arguments from the command line
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_path> <output_path> <density_path>" << std::endl;
        return 1;
    }

    std::string graph_path = argv[1];
    std::string output_path = argv[2];
    std::string density_path = argv[3];

    std::cout << "Graph: " << graph_path << "\nOutput: " << output_path << "\nDensity: " << density_path << std::endl;

    // Configuration
    int accuracy = 1000;
    int max_iter = 100;

    // Convert configuration to strings for argv
    std::string acc_str = std::to_string(accuracy);
    std::string iter_str = std::to_string(max_iter);

    // Build the argv array expected by run_pushrelabel
    // Index mapping based on the updated solver:
    // 0: Program Name
    // 1: Accuracy
    // 2: Max Iterations (Mandatory position in the new solver logic)
    // 3: Graph Path
    // 4: Output Path
    // 5: Density Path
    char *argv_fake[6];
    argv_fake[0] = (char *)"pushrelabel";
    argv_fake[1] = (char *)acc_str.c_str();
    argv_fake[2] = (char *)iter_str.c_str();
    argv_fake[3] = (char *)graph_path.c_str();
    argv_fake[4] = (char *)output_path.c_str();
    argv_fake[5] = (char *)density_path.c_str();

    // Call the solver with argc = 6
    run_pushrelabel(6, argv_fake);

    return 0;
}