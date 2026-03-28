#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace soc::gpu {

class JsonValue {
public:
    enum class Type {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object,
    };

    using Array = std::vector<JsonValue>;
    using Object = std::unordered_map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(bool bool_value);
    explicit JsonValue(double number_value);
    explicit JsonValue(std::string string_value);
    explicit JsonValue(Array array_value);
    explicit JsonValue(Object object_value);

    Type type() const;

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;

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
    bool bool_value_;
    double number_value_;
    std::string string_value_;
    Array array_value_;
    Object object_value_;
};

}  // namespace soc::gpu