#include "header/runtime_pipeline.h"

#include "header/startup_validator.h"

RuntimeBundle RuntimePipeline::LoadBundle(const std::string& manifest_path) {
    const ManifestData manifest = ManifestLoader::LoadFromFile(manifest_path);
    const TokenizerRuntimeData tokenizer_runtime = TokenizerRuntimeLoader::LoadFromFile(manifest.tokenizer_runtime_file);
    StartupValidator::ValidateManifestAndTokenizer(manifest, tokenizer_runtime);
    return RuntimeBundle{manifest, tokenizer_runtime, QwenModelLoader::LoadModel(manifest)};
}

GenerationSession RuntimePipeline::CreateSession(const RuntimeBundle& bundle, const RuntimeGenerationOptions& options) {
    return GenerationSession(
        bundle.model,
        TokenizerRuntime(bundle.tokenizer_runtime),
        Sampler(options.sampler),
        options.max_sequence_length);
}

GenerationResult RuntimePipeline::Generate(const std::string& manifest_path, const std::string& prompt, const RuntimeGenerationOptions& options) {
    const RuntimeBundle bundle = LoadBundle(manifest_path);
    GenerationSession session = CreateSession(bundle, options);
    return session.Generate(prompt, options.max_new_tokens, options.eos_token_id);
}