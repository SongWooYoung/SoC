#include "header/json_parser.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
class ParserState {
public:
    explicit ParserState(const std::string& text) : text_(text), index_(0) {}

    JsonValue ParseValue() {
        SkipWhitespace();
        if (index_ >= text_.size()) {
            throw std::runtime_error("unexpected end of JSON input");
        }

        const char current = text_[index_];
        if (current == 'n') {
            ConsumeLiteral("null");
            return JsonValue();
        }
        if (current == 't') {
            ConsumeLiteral("true");
            return JsonValue(true);
        }
        if (current == 'f') {
            ConsumeLiteral("false");
            return JsonValue(false);
        }
        if (current == '"') {
            return JsonValue(ParseString());
        }
        if (current == '[') {
            return JsonValue(ParseArray());
        }
        if (current == '{') {
            return JsonValue(ParseObject());
        }
        if (current == '-' || std::isdigit(static_cast<unsigned char>(current)) != 0) {
            return JsonValue(ParseNumber());
        }

        throw std::runtime_error("unexpected token in JSON input");
    }

    void EnsureFullyConsumed() {
        SkipWhitespace();
        if (index_ != text_.size()) {
            throw std::runtime_error("unexpected trailing JSON content");
        }
    }

private:
    void SkipWhitespace() {
        while (index_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[index_])) != 0) {
            ++index_;
        }
    }

    void ConsumeLiteral(const char* literal) {
        std::size_t literal_index = 0;
        while (literal[literal_index] != '\0') {
            if (index_ >= text_.size() || text_[index_] != literal[literal_index]) {
                throw std::runtime_error("invalid JSON literal");
            }
            ++index_;
            ++literal_index;
        }
    }

    std::string ParseString() {
        if (text_[index_] != '"') {
            throw std::runtime_error("expected JSON string");
        }

        ++index_;
        std::string result;

        while (index_ < text_.size()) {
            const char current = text_[index_++];
            if (current == '"') {
                return result;
            }
            if (current == '\\') {
                if (index_ >= text_.size()) {
                    throw std::runtime_error("invalid JSON escape sequence");
                }

                const char escaped = text_[index_++];
                switch (escaped) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    case 'u': {
                        if (index_ + 4 > text_.size()) {
                            throw std::runtime_error("invalid unicode escape in JSON string");
                        }

                        const std::string hex = text_.substr(index_, 4);
                        const int code_point = std::stoi(hex, nullptr, 16);
                        index_ += 4;

                        if (code_point <= 0x7F) {
                            result.push_back(static_cast<char>(code_point));
                        } else if (code_point <= 0x7FF) {
                            result.push_back(static_cast<char>(0xC0 | ((code_point >> 6) & 0x1F)));
                            result.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                        } else {
                            result.push_back(static_cast<char>(0xE0 | ((code_point >> 12) & 0x0F)));
                            result.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
                            result.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("unsupported JSON escape sequence");
                }
                continue;
            }

            result.push_back(current);
        }

        throw std::runtime_error("unterminated JSON string");
    }

    JsonValue::Array ParseArray() {
        if (text_[index_] != '[') {
            throw std::runtime_error("expected JSON array");
        }

        ++index_;
        JsonValue::Array array;
        SkipWhitespace();
        if (index_ < text_.size() && text_[index_] == ']') {
            ++index_;
            return array;
        }

        while (true) {
            array.push_back(ParseValue());
            SkipWhitespace();
            if (index_ >= text_.size()) {
                throw std::runtime_error("unterminated JSON array");
            }
            if (text_[index_] == ']') {
                ++index_;
                return array;
            }
            if (text_[index_] != ',') {
                throw std::runtime_error("expected comma in JSON array");
            }
            ++index_;
        }
    }

    JsonValue::Object ParseObject() {
        if (text_[index_] != '{') {
            throw std::runtime_error("expected JSON object");
        }

        ++index_;
        JsonValue::Object object;
        SkipWhitespace();
        if (index_ < text_.size() && text_[index_] == '}') {
            ++index_;
            return object;
        }

        while (true) {
            SkipWhitespace();
            const std::string key = ParseString();
            SkipWhitespace();
            if (index_ >= text_.size() || text_[index_] != ':') {
                throw std::runtime_error("expected colon in JSON object");
            }
            ++index_;
            object.emplace(key, ParseValue());
            SkipWhitespace();
            if (index_ >= text_.size()) {
                throw std::runtime_error("unterminated JSON object");
            }
            if (text_[index_] == '}') {
                ++index_;
                return object;
            }
            if (text_[index_] != ',') {
                throw std::runtime_error("expected comma in JSON object");
            }
            ++index_;
        }
    }

    double ParseNumber() {
        const std::size_t start = index_;
        if (text_[index_] == '-') {
            ++index_;
        }

        while (index_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[index_])) != 0) {
            ++index_;
        }

        if (index_ < text_.size() && text_[index_] == '.') {
            ++index_;
            while (index_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[index_])) != 0) {
                ++index_;
            }
        }

        if (index_ < text_.size() && (text_[index_] == 'e' || text_[index_] == 'E')) {
            ++index_;
            if (index_ < text_.size() && (text_[index_] == '+' || text_[index_] == '-')) {
                ++index_;
            }
            while (index_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[index_])) != 0) {
                ++index_;
            }
        }

        return std::stod(text_.substr(start, index_ - start));
    }

    const std::string& text_;
    std::size_t index_;
};
}

JsonValue::JsonValue() : type_(Type::Null), bool_value_(false), number_value_(0.0) {}

JsonValue::JsonValue(bool bool_value) : type_(Type::Bool), bool_value_(bool_value), number_value_(0.0) {}

JsonValue::JsonValue(double number_value) : type_(Type::Number), bool_value_(false), number_value_(number_value) {}

JsonValue::JsonValue(std::string string_value)
    : type_(Type::String), bool_value_(false), number_value_(0.0), string_value_(std::move(string_value)) {}

JsonValue::JsonValue(Array array_value)
    : type_(Type::Array), bool_value_(false), number_value_(0.0), array_value_(std::move(array_value)) {}

JsonValue::JsonValue(Object object_value)
    : type_(Type::Object), bool_value_(false), number_value_(0.0), object_value_(std::move(object_value)) {}

JsonValue::Type JsonValue::type() const {
    return type_;
}

bool JsonValue::is_null() const {
    return type_ == Type::Null;
}

bool JsonValue::is_bool() const {
    return type_ == Type::Bool;
}

bool JsonValue::is_number() const {
    return type_ == Type::Number;
}

bool JsonValue::is_string() const {
    return type_ == Type::String;
}

bool JsonValue::is_array() const {
    return type_ == Type::Array;
}

bool JsonValue::is_object() const {
    return type_ == Type::Object;
}

bool JsonValue::as_bool() const {
    if (!is_bool()) {
        throw std::runtime_error("JSON value is not a bool");
    }
    return bool_value_;
}

double JsonValue::as_number() const {
    if (!is_number()) {
        throw std::runtime_error("JSON value is not a number");
    }
    return number_value_;
}

int JsonValue::as_int() const {
    return static_cast<int>(std::llround(as_number()));
}

std::int64_t JsonValue::as_int64() const {
    return static_cast<std::int64_t>(std::llround(as_number()));
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) {
        throw std::runtime_error("JSON value is not a string");
    }
    return string_value_;
}

const JsonValue::Array& JsonValue::as_array() const {
    if (!is_array()) {
        throw std::runtime_error("JSON value is not an array");
    }
    return array_value_;
}

const JsonValue::Object& JsonValue::as_object() const {
    if (!is_object()) {
        throw std::runtime_error("JSON value is not an object");
    }
    return object_value_;
}

bool JsonValue::contains(const std::string& key) const {
    if (!is_object()) {
        return false;
    }
    return object_value_.find(key) != object_value_.end();
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const auto iterator = as_object().find(key);
    if (iterator == as_object().end()) {
        throw std::runtime_error("missing JSON object key: " + key);
    }
    return iterator->second;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (!is_object()) {
        return nullptr;
    }
    const auto iterator = object_value_.find(key);
    if (iterator == object_value_.end()) {
        return nullptr;
    }
    return &iterator->second;
}

JsonValue JsonParser::Parse(const std::string& text) {
    ParserState parser(text);
    JsonValue value = parser.ParseValue();
    parser.EnsureFullyConsumed();
    return value;
}

JsonValue JsonParser::ParseFile(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open JSON file: " + path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return Parse(buffer.str());
}