#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <thread>
#include <functional>
#include <nlohmann/json.hpp>
#include "whisper.h"

// Create alias for json library
using json = nlohmann::json;

// Forward declarations
class Logger;
struct whisper_params;

// Helper structure for token probabilities
struct token_prob {
    std::string text;
    float p;
};

// Command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool json_output   = false; // output in JSON format
    bool print_tokens  = false; // include tokens in output
    bool replay        = false; // replay mode (read from JSONL file)
    std::string replay_file;  // path to JSONL file for replay

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

// Function to convert whisper_params to JSON
json create_params_json(const whisper_params& params);

// Function declarations for parameter handling
bool whisper_params_parse(int argc, char** argv, whisper_params& params);
void whisper_print_usage(int argc, char** argv, const whisper_params& params);