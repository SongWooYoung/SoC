#pragma once

#include <mlx/mlx.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qwen3_5_mlx {
namespace mx = mlx::core;

struct WeightLoader {
    std::unordered_map<std::string, mx::array> tensors;

    void set(const std::string& key, const mx::array& value) {
        tensors.insert_or_assign(key, value);
    }

    bool contains(const std::string& key) const {
        return tensors.find(key) != tensors.end();
    }

    const mx::array& at(const std::string& key) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            throw std::out_of_range("Missing weight tensor: " + key);
        }
        return it->second;
    }

    std::optional<mx::array> maybe(const std::string& key) const {
        auto it = tensors.find(key);
        if (it == tensors.end()) {
            return std::nullopt;
        }
        return it->second;
    }
};

} // namespace qwen3_5_mlx