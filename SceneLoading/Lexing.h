#pragma once

#include <string>
#include <set>
#include <unordered_set>
#include <iostream>
#include "../Utils/GpuCommons.h"
#include "../Utils/Utils.h"
#include "../Utils/Variant.h"
#include <vector> 
#include <memory>
#include <filesystem>

enum class TokenType{
    KeyWord, String,Num, LeftSquareBracket, RightSquareBracket
};

class TokenBase {
public:
    TokenType type;
    virtual std::string print() = 0;
};

class TokenBuf;


class KeyWordToken: public TokenBase{
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

class StringToken: public TokenBase{
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

class NumToken: public TokenBase{
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

class LeftSquareBracketToken: public TokenBase{
public:
    LeftSquareBracketToken(){
        type = TokenType::LeftSquareBracket;
    }
    virtual std::string print() override{
        return "[";
    }
};

class RightSquareBracketToken: public TokenBase{
public:
    RightSquareBracketToken(){
        type = TokenType::RightSquareBracket;
    }
    virtual std::string print() override{
        return "]";
    }
};




using TokenVariant = Variant<KeyWordToken,StringToken,NumToken,LeftSquareBracketToken,RightSquareBracketToken>;

class Token : public TokenVariant {
public:

	Token() {}

	template<typename V>
	Token(const V& v) :TokenVariant(v) {}

	Token(const Token& other) : TokenVariant(other.value) {}

	__device__
	Token& operator=(const Token& other) {
		value = other.value;
		return *this;
	}

    std::string print() {
        auto visitor = [&](auto& arg) -> std::string {
            using T = typename std::remove_reference<decltype(arg)>::type;
            if constexpr (std::is_base_of<TokenBase, typename T>::value) {
                return arg.T::print();
            }
            else {
                SIGNAL_VARIANT_ERROR;
            }
        };
        return visit(visitor);
    }

	TokenType type() {
		auto visitor = [&](auto& arg) -> TokenType {
			using T = typename std::remove_reference<decltype(arg)>::type;
			if constexpr (std::is_base_of<TokenBase, typename T>::value) {
				return arg.type;
			}
			else {
				SIGNAL_VARIANT_ERROR;
			}
		};
		return visit(visitor);
	}
};



struct TokenBuf {
    std::vector<Token> tokens;
    int currentIndex = 0;

    Token* peek(int offset = 0) {
        if (currentIndex + offset >= tokens.size()) {
            SIGNAL_ERROR("peek out of bounds");
        }
        return &tokens[currentIndex + offset];
    }

    void moveForward() {
        ++currentIndex;
    }

    template <typename T>
    T* checkAndPop() {
        Token* thisToken = peek();
        if (thisToken->is<T>()) {
            ++currentIndex;
            return thisToken->get<T>();
        }
        SIGNAL_ERROR((std::string("Token checkAndPop failed. Token index: ") + std::to_string(currentIndex) + "." + thisToken->print()).c_str());
    }

    void insertHere(const TokenBuf& buf) {
        tokens.insert(tokens.begin() + currentIndex, buf.tokens.begin(),buf.tokens.end());
    }
};

TokenBuf runLexing(const std::filesystem::path& inputPath);
