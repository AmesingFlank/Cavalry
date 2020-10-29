#pragma once

#include <string>
#include <set>
#include <unordered_set>
#include <iostream>
#include "../Utils/GpuCommons.h"
#include "../Utils/Utils.h"
#include <vector> 
#include <memory>
#include <filesystem>

enum class TokenType{
    KeyWord, String,Num, LeftSquareBracket, RightSquareBracket
};

class Token {
public:
    TokenType type;
    virtual std::string print() = 0;
};

struct TokenBuf {
    std::vector<std::shared_ptr<Token>> tokens;
    int currentIndex = 0;

    std::shared_ptr<Token> peek(int offset = 0){
        if (currentIndex+offset >= tokens.size()) {
            SIGNAL_ERROR("peek out of bounds");
        }
        return tokens[currentIndex+offset];
    }

    void moveForward(){
        ++currentIndex;
    }

    template <typename T>
    std::shared_ptr<T> checkAndPop(){
        std::shared_ptr<Token> thisToken = peek();
        std::shared_ptr<T> casted = std::dynamic_pointer_cast<T>(thisToken);
        if(casted){
            ++currentIndex;
            return casted;
        }
        SIGNAL_ERROR((std::string("Token checkAndPop failed. Token index: ")+std::to_string(currentIndex)+ "." + thisToken->print()).c_str());
    }

    void insertHere(const TokenBuf& buf) {
        for (auto& token : buf.tokens) {
            tokens.insert(tokens.begin()+currentIndex, token);
        }
    }
};

TokenBuf runLexing(const std::filesystem::path& inputPath);


class KeyWordToken: public Token{
public:
    std::string word;

    static const std::unordered_set<std::string> recognized;

    KeyWordToken(const std::string& word_):word(word_){
        if(recognized.find(word)==recognized.end()){
            SIGNAL_ERROR((std::string("Unrecognized Keyword: ")+word).c_str());
        }
        type = TokenType::KeyWord;
    }

    static void read(const std::string& input, int& pos, TokenBuf& result);

    virtual std::string print() override{
        return word;
    }
};

class StringToken: public Token{
public:
    std::vector<std::string> words;
    std::string all;
    StringToken(const std::string& raw):words(splitString(raw," ")),all(raw){
        type = TokenType::String;
    }
    static void read(const std::string& input, int& pos, TokenBuf& result);
    virtual std::string print() override{
        std::string result = "\"";
        result += all;
        result += "\"";
        return result;
    }
};

class NumToken: public Token{
public:
    float value;
    NumToken(float value_):value(value_){
        type = TokenType::Num;
    }
    static void read(const std::string& input, int& pos, TokenBuf& result);
    virtual std::string print() override{
        return std::to_string(value);
    }
};

class LeftSquareBracketToken: public Token{
public:
    LeftSquareBracketToken(){
        type = TokenType::LeftSquareBracket;
    }
    virtual std::string print() override{
        return "[";
    }
};

class RightSquareBracketToken: public Token{
public:
    RightSquareBracketToken(){
        type = TokenType::RightSquareBracket;
    }
    virtual std::string print() override{
        return "]";
    }
};



