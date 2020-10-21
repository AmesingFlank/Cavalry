#pragma once

#include <string>
#include <set>
#include <unordered_set>
#include <iostream>
#include "../Utils/GpuCommons.h"
#include "../Utils/Utils.h"
#include <vector> 
#include <memory>

enum class TokenType{
    KeyWord, String,Num, LeftSquareBracket, RightSquareBracket, Comma
};

class Token {
public:
    TokenType type;
};

struct TokenBuf {
    std::vector<std::shared_ptr<Token>> tokens;
    int currentIndex = 0;
};

TokenBuf runLexing(const std::string& input);


class KeyWordToken: public Token{
public:
    std::string word;
    KeyWordToken(const std::string& word_):word(word_){
        std::unordered_set<std::string> recognized = 
        {""};
        if(recognized.find(word)==recognized.end()){
            SIGNAL_ERROR((std::string("Unrecognized Keyword: ")+word).c_str());
        }
        type = TokenType::KeyWord;
    }
    static void read(const std::string& input, int& pos, TokenBuf& result);
};

class StringToken: public Token{
public:
    std::vector<std::string> words;
    StringToken(const std::string& raw):words(splitString(raw," ")){
        type = TokenType::String;
    }
    static void read(const std::string& input, int& pos, TokenBuf& result);
};

class NumToken: public Token{
public:
    float value;
    NumToken(float value_):value(value_){
        type = TokenType::Num;
    }
    static void read(const std::string& input, int& pos, TokenBuf& result);
};

class LeftSquareBracketToken: public Token{
public:
    LeftSquareBracketToken(){
        type = TokenType::LeftSquareBracket;
    }
};

class RightSquareBracketToken: public Token{
public:
    RightSquareBracketToken(){
        type = TokenType::RightSquareBracket;
    }
};

class CommaToken: public Token{
public:
    CommaToken(){
        type = TokenType::Comma;
    }
};


