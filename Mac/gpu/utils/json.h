#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    using Array = std::vector<JsonValue>;
    using Object = std::unordered_map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(bool v);
    explicit JsonValue(double v);
    explicit JsonValue(std::string v);
    explicit JsonValue(Array v);
    explicit JsonValue(Object v);

    Type type() const { return type_; }

    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool() const;
    double as_number() const;
    int as_int() const;
    std::int64_t as_int64() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    const Object& as_object() const;

    bool contains(const std::string& key) const;
    const JsonValue& at(const std::string& key) const;
    const JsonValue* find(const std::string& key) const;

private:
    Type type_;
    bool bool_value_{};
    double number_value_{};
    std::string string_value_;
    Array array_value_;
    Object object_value_;
};

class JsonParser {
public:
    static JsonValue parse(const std::string& text);
    static JsonValue parse_file(const std::string& path);
};
