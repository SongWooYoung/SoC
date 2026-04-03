#include "json.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

// -- JsonValue ----------------------------------------------------------------

JsonValue::JsonValue() : type_(Type::Null) {}
JsonValue::JsonValue(bool v) : type_(Type::Bool), bool_value_(v) {}
JsonValue::JsonValue(double v) : type_(Type::Number), number_value_(v) {}
JsonValue::JsonValue(std::string v) : type_(Type::String), string_value_(std::move(v)) {}
JsonValue::JsonValue(Array v) : type_(Type::Array), array_value_(std::move(v)) {}
JsonValue::JsonValue(Object v) : type_(Type::Object), object_value_(std::move(v)) {}

bool JsonValue::as_bool() const {
    if (!is_bool()) throw std::runtime_error("JSON value is not a bool");
    return bool_value_;
}

double JsonValue::as_number() const {
    if (!is_number()) throw std::runtime_error("JSON value is not a number");
    return number_value_;
}

int JsonValue::as_int() const {
    return static_cast<int>(std::llround(as_number()));
}

std::int64_t JsonValue::as_int64() const {
    return static_cast<std::int64_t>(std::llround(as_number()));
}

const std::string& JsonValue::as_string() const {
    if (!is_string()) throw std::runtime_error("JSON value is not a string");
    return string_value_;
}

const JsonValue::Array& JsonValue::as_array() const {
    if (!is_array()) throw std::runtime_error("JSON value is not an array");
    return array_value_;
}

const JsonValue::Object& JsonValue::as_object() const {
    if (!is_object()) throw std::runtime_error("JSON value is not an object");
    return object_value_;
}

bool JsonValue::contains(const std::string& key) const {
    if (!is_object()) return false;
    return object_value_.find(key) != object_value_.end();
}

const JsonValue& JsonValue::at(const std::string& key) const {
    const auto it = as_object().find(key);
    if (it == as_object().end())
        throw std::runtime_error("missing JSON key: " + key);
    return it->second;
}

const JsonValue* JsonValue::find(const std::string& key) const {
    if (!is_object()) return nullptr;
    const auto it = object_value_.find(key);
    if (it == object_value_.end()) return nullptr;
    return &it->second;
}

// -- Parser -------------------------------------------------------------------

namespace {

class ParserState {
public:
    explicit ParserState(const std::string& text) : text_(text), pos_(0) {}

    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");

        switch (text_[pos_]) {
            case 'n': consume("null"); return JsonValue();
            case 't': consume("true"); return JsonValue(true);
            case 'f': consume("false"); return JsonValue(false);
            case '"': return JsonValue(parse_string());
            case '[': return JsonValue(parse_array());
            case '{': return JsonValue(parse_object());
            default:
                if (text_[pos_] == '-' || std::isdigit(static_cast<unsigned char>(text_[pos_])))
                    return JsonValue(parse_number());
                throw std::runtime_error("unexpected token in JSON");
        }
    }

    void ensure_fully_consumed() {
        skip_ws();
        if (pos_ != text_.size())
            throw std::runtime_error("trailing JSON content");
    }

private:
    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
            ++pos_;
    }

    void consume(const char* lit) {
        for (std::size_t i = 0; lit[i]; ++i) {
            if (pos_ >= text_.size() || text_[pos_] != lit[i])
                throw std::runtime_error("invalid JSON literal");
            ++pos_;
        }
    }

    std::string parse_string() {
        if (text_[pos_] != '"') throw std::runtime_error("expected '\"'");
        ++pos_;
        std::string result;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') return result;
            if (c == '\\') {
                if (pos_ >= text_.size()) throw std::runtime_error("invalid escape");
                char esc = text_[pos_++];
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) throw std::runtime_error("invalid \\u escape");
                        int cp = std::stoi(text_.substr(pos_, 4), nullptr, 16);
                        pos_ += 4;
                        if (cp <= 0x7F) {
                            result += static_cast<char>(cp);
                        } else if (cp <= 0x7FF) {
                            result += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            result += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            result += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: throw std::runtime_error("unsupported escape");
                }
                continue;
            }
            result += c;
        }
        throw std::runtime_error("unterminated string");
    }

    JsonValue::Array parse_array() {
        ++pos_; // '['
        JsonValue::Array arr;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == ']') { ++pos_; return arr; }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (pos_ >= text_.size()) throw std::runtime_error("unterminated array");
            if (text_[pos_] == ']') { ++pos_; return arr; }
            if (text_[pos_] != ',') throw std::runtime_error("expected ',' in array");
            ++pos_;
        }
    }

    JsonValue::Object parse_object() {
        ++pos_; // '{'
        JsonValue::Object obj;
        skip_ws();
        if (pos_ < text_.size() && text_[pos_] == '}') { ++pos_; return obj; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (pos_ >= text_.size() || text_[pos_] != ':')
                throw std::runtime_error("expected ':' in object");
            ++pos_;
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            if (pos_ >= text_.size()) throw std::runtime_error("unterminated object");
            if (text_[pos_] == '}') { ++pos_; return obj; }
            if (text_[pos_] != ',') throw std::runtime_error("expected ',' in object");
            ++pos_;
        }
    }

    double parse_number() {
        std::size_t start = pos_;
        if (text_[pos_] == '-') ++pos_;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        return std::stod(text_.substr(start, pos_ - start));
    }

    const std::string& text_;
    std::size_t pos_;
};

} // namespace

JsonValue JsonParser::parse(const std::string& text) {
    ParserState p(text);
    JsonValue v = p.parse_value();
    p.ensure_fully_consumed();
    return v;
}

JsonValue JsonParser::parse_file(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("failed to open: " + path);
    std::ostringstream buf;
    buf << stream.rdbuf();
    return parse(buf.str());
}
