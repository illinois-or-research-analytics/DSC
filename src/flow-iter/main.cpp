#include "pushrelabel.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <graph_path> <output_path> <density_path> [<density_threshold>]" << std::endl;
        return 1;
    }
    else if (argc < 5)
    {
        std::cout << "Density threshold not provided. Using default value of 0.0." << std::endl;
    }

    std::string graph_path = argv[1];
    std::string output_path = argv[2];
    std::string density_path = argv[3];
    double density_threshold = (argc >= 5) ? std::stod(argv[4]) : 0.0;

    std::cout << "Graph path: " << graph_path << std::endl;
    std::cout << "Output path: " << output_path << std::endl;
    std::cout << "Density path: " << density_path << std::endl;
    std::cout << "Density threshold: " << density_threshold << std::endl;

    int accuracy = 1000;
    int max_iter = 100;

    std::string acc_str = std::to_string(accuracy);
    std::string iter_str = std::to_string(max_iter);
    std::string density_threshold_str = std::to_string(density_threshold);

    const int argc_fake = 7;
    char *argv_fake[argc_fake];

    argv_fake[0] = (char *)"pushrelabel";
    argv_fake[1] = (char *)acc_str.c_str();
    argv_fake[2] = (char *)iter_str.c_str();
    argv_fake[3] = (char *)graph_path.c_str();
    argv_fake[4] = (char *)output_path.c_str();
    argv_fake[5] = (char *)density_path.c_str();
    argv_fake[6] = (char *)density_threshold_str.c_str();

    run_pushrelabel(argc_fake, argv_fake);

    return 0;
}