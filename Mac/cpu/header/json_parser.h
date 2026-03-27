#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <string>

#include "header/json_value.h"

class JsonParser {
public:
    static JsonValue Parse(const std::string& text);
    static JsonValue ParseFile(const std::string& path);
};

#endif