/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _PUMA_TEST_LSTM_LAYER_
#define _PUMA_TEST_LSTM_LAYER_

static std::vector< std::vector<Vector> > depthwise_conv(Model model, 
                                          std::string layerName, 
                                          unsigned int k_size_x, 
                                          unsigned int k_size_y, 
                                          unsigned int in_size_x, 
                                          unsigned int in_size_y, 
                                          unsigned int n_channels,
                                          unsigned int stride, 
                                          unsigned int out_size_x, 
                                          unsigned int out_size_y) {

    // Create random image input
    unsigned int height = k_size_x * k_size_y;
    
    // Matrix columns
    unsigned int vectorLength = out_size_x * out_size_y;

    std::vector< std::vector<Vector> > inputImageTiled(n_channels);
    for(unsigned int c = 0; c < n_channels; c++) {
        inputImageTiled[c].resize(vectorLength);
        for(unsigned int i = 0; i < vectorLength; ++i) {
            inputImageTiled[c][i] = InputVector::create(model, layerName + std::to_string(i), height);
        }
    }                                    
    
    // Create kernel for depthwise conv
    // Kernels flattened
    std::vector<ConstantMatrix> kernel(n_channels);
    for(unsigned int i = 0; i < n_channels; ++i) {
        kernel[i] = ConstantMatrix::create(model, layerName + "k" + std::to_string(i), height, 1);
    }

    std::vector< std::vector<Vector> > outputFlattenedImage(n_channels);
    std::vector< std::vector<OutputVector> > outputOutFlattenedImage(n_channels);
    for(unsigned int c = 0; c < n_channels; c++) {
        // Loop across kernel positons
        outputFlattenedImage[c].resize(vectorLength);
        outputOutFlattenedImage[c].resize(vectorLength);
        for(unsigned int imageColumn = 0; imageColumn < vectorLength;  imageColumn++) {
            auto out = OutputVector::create(model,"dws", 1);
            auto compOut = kernel[c] * inputImageTiled[c][imageColumn];
            outputFlattenedImage[c][imageColumn] = compOut;
            out = compOut;
            // outputFlattenedImage[c][imageColumn] = kernel[c] * inputImageTiled[c][imageColumn];
        }
    }

    return  outputFlattenedImage;
}

#endif