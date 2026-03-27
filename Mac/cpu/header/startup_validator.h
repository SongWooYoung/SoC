#ifndef STARTUP_VALIDATOR_H
#define STARTUP_VALIDATOR_H

#include <string>

#include "header/qwen3_config.h"
#include "header/runtime_assets.h"

class StartupValidator {
public:
    static Qwen3Config ValidateManifest(const ManifestData& manifest);
    static void ValidateTokenizerRuntime(const TokenizerRuntimeData& tokenizer_runtime);
    static void ValidateManifestAndTokenizer(const ManifestData& manifest, const TokenizerRuntimeData& tokenizer_runtime);

private:
    static int FindTokenIdByContent(const TokenizerRuntimeData& tokenizer_runtime, const std::string& token_content);
};

#endif