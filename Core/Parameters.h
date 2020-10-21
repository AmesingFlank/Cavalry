#pragma once

#include <unordered_map>

struct Parameters{

    std::unordered_map<std::string,std::string> strings;
    std::unordered_map<std::string,std::vector<std::string>> stringLists;
    std::unordered_map<std::string,float> nums;
    std::unordered_map<std::string,std::vector<float>> numLists;

};