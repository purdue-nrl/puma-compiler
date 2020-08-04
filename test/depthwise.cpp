/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <string>
#include <vector>
#include <iostream>

#include "puma.h"
#include "depthwise.h"
// using namespace std;
int main(int argc, char** argv) {

//    Model model = Model::create("conv3-layer");

    // Process parameter
    unsigned int in_size_x=9 ; 
    unsigned int in_size_y=9 ; 
    unsigned int n_channels=16 ;
    unsigned int out_channels=16 ;
    unsigned int k_size_x=3 ;
    unsigned int k_size_y=3 ;
    unsigned int padding=1 ;
    unsigned int stride=1 ;

    if(argc == 10) {
        in_size_x = atoi(argv[1]);
        in_size_y = atoi(argv[2]);
        n_channels = atoi(argv[3]);
        k_size_x = atoi(argv[5]);
        k_size_y = atoi(argv[6]);
        padding = atoi(argv[7]);
        stride = atoi(argv[8]);
    }    
    std::string str=std::string("depthwise") + argv[9];
    Model model = Model::create(str);
    // Model model = Model::create(str);
   
    unsigned int out_size_x = (in_size_x - k_size_x + 2*padding)/stride + 1;
    unsigned int out_size_y =  (in_size_y - k_size_y + 2*padding)/stride + 1;

    // Layer
    auto out = depthwise_conv(model, "", k_size_x, k_size_y, in_size_x, in_size_y, n_channels, stride, out_size_x, out_size_y);
    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}

