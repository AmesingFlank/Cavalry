#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


inline std::string readTextFile(const std::string& path) {
    std::string result;
    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    file.open(path);

    std::stringstream stream;

    stream << file.rdbuf();

    result = stream.str();
    return result;
}

inline std::vector<std::string> splitString(const std::string& s, const std::string& delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

inline std::string getFileNamePostfix(const std::string& filename) {
    auto parts = splitString(filename, ".");
    if (parts.size() == 0) {
        return "";
    }
    return parts[parts.size() - 1];
}

inline bool isLetter(char c){
    return ('a'<=c && c<='z') ||  ('A'<=c && c<='Z');
}

inline bool isDigit(char c){
    return ('0'<=c && c<='9');
}


inline bool endsWith(const std::string& word,const std::string& end) {
    if (word.size() < end.size()) return false;
    return word.substr(word.size() - end.size(), end.size()) == end;
}