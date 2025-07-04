
#ifndef _RFIMARKER_H
#define _RFIMARKER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>


template <typename T>
class RfiMarker {
public:
    RfiMarker();
    RfiMarker(const char* mask_file); // Constructor that takes a mask file
    ~RfiMarker() = default;

    std::vector<int> bad_channels; // Vector to store bad channels

    // Method to load the RFI mask
    void load_mask(const char* mask_file);

    void mark_rfi(T* data, uint num_channels, uint num_samples);
};



template <typename T>
RfiMarker<T>::RfiMarker() {
    load_mask("mask.txt"); // Default mask file
}

template <typename T>
RfiMarker<T>::RfiMarker(const char* mask_file) {
    load_mask(mask_file); // Load the mask from the provided file
}

template <typename T>
void RfiMarker<T>::mark_rfi(T* data, uint num_channels, uint num_samples) {
    // Iterate through the bad channels and mark them in the data
    #pragma omp parallel for
    for (int chan : bad_channels) {
        if (chan >= 0 && chan < num_channels) {
            #pragma omp simd
            for (uint sample = 0; sample < num_samples; ++sample) {
                // Set the data for the bad channel to zero
                data[sample * num_channels + chan] = 0;
            }
        } else {
            std::cerr << "Warning: Bad channel index " << chan << " out of range." << std::endl;
        }
    }
}

template <typename T>
void RfiMarker<T>::load_mask(const char* mask_file) {
    // open the mask file
    std::ifstream file(mask_file);
    if (!file.is_open()) {
        std::cerr << "Error opening mask file: " << mask_file << std::endl
                    << "Please check the file path and try again." << std::endl;
        return;
    }
    // if tempty continue
    if (file.peek() == std::ifstream::traits_type::eof()) {
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int chan;
        while (ss >> chan) {
            bad_channels.push_back(chan);
        }
    }
    file.close();
}
#endif //_RFIMARKER_H