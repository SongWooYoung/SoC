#pragma once

// Model Registry — swap model implementations via a single include.
//
// Usage:
//   #include "models/registry.h"
//   auto model = ModelRegistry::load("qwen3_5_py_cpp", bundle, config);
//   auto tokens = model->generate(prompt_ids, 64, 248046);
//
// To add a new model:
//   1. Create models/<name>/ with modeling.h exposing a ForCausalLM struct
//   2. Add a loader function below
//   3. Register it in ModelRegistry::load()

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// Abstract model interface
// ═══════════════════════════════════════════════════════════════════════════

struct IForCausalLM {
    virtual ~IForCausalLM() = default;

    virtual int get_vocab_size() const = 0;

    // Greedy generation: returns full sequence (prompt + generated)
    virtual std::vector<int> generate(
        const std::vector<int>& prompt_ids,
        int max_new_tokens,
        int eos_token_id) const = 0;
};

// ═══════════════════════════════════════════════════════════════════════════
// Concrete adapters — one per model implementation
// ═══════════════════════════════════════════════════════════════════════════

// --- qwen3_5_py_cpp (PyTorch → C++ port) ---
#include "models/qwen3_5_py_cpp/modeling.h"

struct Qwen3_5PyCppAdapter : public IForCausalLM {
    Qwen3_5ForCausalLM inner;

    void load(const SafetensorsBundle& bundle,
              const std::string& prefix,
              const Qwen3_5Config& config) {
        inner.load(bundle, prefix, config);
    }

    int get_vocab_size() const override { return inner.vocab_size; }

    std::vector<int> generate(
        const std::vector<int>& prompt_ids,
        int max_new_tokens,
        int eos_token_id) const override {
        return inner.generate(prompt_ids, max_new_tokens, eos_token_id);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Registry
// ═══════════════════════════════════════════════════════════════════════════

namespace ModelRegistry {

inline std::vector<std::string> available_models() {
    return {"qwen3_5_py_cpp"};
}

// Load a model by name. Returns owning pointer.
// Caller provides the SafetensorsBundle and config already parsed.
inline std::unique_ptr<IForCausalLM> load_qwen3_5_py_cpp(
    const SafetensorsBundle& bundle,
    const std::string& prefix,
    const Qwen3_5Config& config) {
    auto m = std::make_unique<Qwen3_5PyCppAdapter>();
    m->load(bundle, prefix, config);
    return m;
}

} // namespace ModelRegistry
