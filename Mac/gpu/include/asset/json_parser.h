#pragma once

#include <string>

#include "asset/json_value.h"

namespace soc::gpu {

class JsonParser {
public:
    static JsonValue Parse(const std::string& text);
    static JsonValue ParseFile(const std::string& path);
};

}  // namespace soc::gpu