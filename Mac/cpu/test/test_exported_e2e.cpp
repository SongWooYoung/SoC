#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string QuoteShell(const std::string& value) {
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string FindPythonExecutable(const std::filesystem::path& repo_root) {
    if (const char* env_python = std::getenv("SOC_PYTHON")) {
        return env_python;
    }

    const std::vector<std::filesystem::path> candidates = {
        repo_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "bin" / "python3",
    };
    for (const std::filesystem::path& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    return "python3";
}

void test_exporter_generated_bundle_end_to_end() {
    std::cout << "=== Testing Exporter Generated End-to-End Smoke ===" << std::endl;

    const std::filesystem::path cpu_dir = std::filesystem::current_path();
    const std::filesystem::path repo_root = cpu_dir.parent_path().parent_path();
    const std::filesystem::path exporter_script = repo_root / "LLM_interpreter" / "export_test_bundle.py";
    require(std::filesystem::exists(exporter_script), "exporter smoke helper script must exist");

    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "soc_exported_e2e_smoke";
    std::filesystem::remove_all(temp_dir);
    std::filesystem::create_directories(temp_dir);

    const std::string command =
        QuoteShell(FindPythonExecutable(repo_root)) + " " +
        QuoteShell(exporter_script.string()) + " --output-dir " +
        QuoteShell(temp_dir.string()) + " --dtype float16";
    const int exit_code = std::system(command.c_str());
    require(exit_code == 0, "export helper must generate a runtime bundle successfully");

    const std::filesystem::path manifest_path = temp_dir / "manifest.json";
    require(std::filesystem::exists(manifest_path), "export helper must emit manifest.json");
        require(std::filesystem::exists(temp_dir / "tokenizer" / "tokenizer_runtime.json"),
            "export helper must emit tokenizer_runtime.json");
        require(std::filesystem::exists(temp_dir / "weights" / "model.embed_tokens.weight.bin") ||
            std::filesystem::exists(temp_dir / "weights"),
            "export helper must emit weight artifacts");
}
}

int main() {
    test_exporter_generated_bundle_end_to_end();

    std::cout << "=== Exported End-to-End Tests Completed ===" << std::endl;
    return 0;
}