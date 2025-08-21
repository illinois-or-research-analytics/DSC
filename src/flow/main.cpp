#include "pushrelabel.h"
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    std::string graph_path = argv[1];
    std::string output_path = argv[2];
    std::string density_path = argv[3];

    std::cout << "output path: " << output_path << std::endl;

    /* // Redirect input file to stdin */
    /* if (freopen(graph_path.c_str(), "r", stdin) == nullptr) */
    /* { */
    /*     std::cerr << "Error: could not open " << graph_path << " for reading." << std::endl; */
    /*     return 1; */
    /* } */
    // pick your accuracy however you like:
    int accuracy = 1000;

    // build a little argv array with program name + accuracy
    std::string acc_str = std::to_string(accuracy);
    char *argv_fake[5];
    argv_fake[0] = (char *)"pushrelabel";
    argv_fake[1] = (char *)acc_str.c_str();
    argv_fake[2] = (char *)graph_path.c_str();
    argv_fake[3] = (char *)output_path.c_str();
    argv_fake[4] = (char *)density_path.c_str();

    // now argc_fake is 2, and argv_fake[1] is accuracy
    run_pushrelabel(5, argv_fake);
    return 0;
}
