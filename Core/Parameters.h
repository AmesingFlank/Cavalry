#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "../Utils/GpuCommons.h"

struct Parameters{

    std::unordered_map<std::string,std::string> strings;
    std::unordered_map<std::string,std::vector<std::string>> stringLists;
    std::unordered_map<std::string,float> nums;
    std::unordered_map<std::string,std::vector<float>> numLists;


    bool hasNum(const std::string& name) const {
        if (nums.find(name) != nums.end()) {
            return true;
        }
        if (numLists.find(name) != numLists.end()) {
            return true;
        }
        return false;
    }

    float getNum(const std::string& name) const{
        if(nums.find(name)!=nums.end()){
            return nums.at(name);
        }
        if (numLists.find(name) != numLists.end()) {
            return numLists.at(name)[0];
        }

        SIGNAL_ERROR((std::string("Params Num field not found: ") + name).c_str());
       
    }


    bool hasString(const std::string& name)const {
        if (strings.find(name) != strings.end()) {
            return true;
        }
        if (stringLists.find(name) != stringLists.end()) {
            return true;
        }
        return false;
    }


    std::string getString(const std::string& name)const {
        if (strings.find(name) != strings.end()) {
            return strings.at(name);
        }
        if(stringLists.find(name)==stringLists.end()){
            return stringLists.at(name)[0];
        }
        SIGNAL_ERROR((std::string("Params String field not found: ") + name).c_str());
    }

    bool hasNumList(const std::string& name)const{
        return numLists.find(name) != numLists.end();
    }

    std::vector<float> getNumList(const std::string& name)const {
        if (numLists.find(name) == numLists.end()) {
            SIGNAL_ERROR((std::string("Params NumList field not found: ") + name).c_str());
        }
        return numLists.at(name);
    }

};



struct ObjectDefinition{
    std::string keyWord;
    std::string objectName;
	Parameters params;
	bool isDefined = false;
};