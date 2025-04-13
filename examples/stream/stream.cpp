// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <functional>
#include <nlohmann/json.hpp>

// Create alias for json library
using json = nlohmann::json;

// Helper function to get token probabilities for a segment
struct token_prob {
    std::string text;
    float p;
};
// Forward declarations
class Logger;
struct whisper_params;

// Function declarations
void output_system_info(whisper_context* ctx, Logger& logger, const whisper_params& params);
json create_params_json(const whisper_params& params);


// command-line parameters
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
json create_params_json(const whisper_params& params) {
    return {
        {"n_threads", params.n_threads},
        {"step_ms", params.step_ms},
        {"length_ms", params.length_ms},
        {"keep_ms", params.keep_ms},
        {"capture_id", params.capture_id},
        {"max_tokens", params.max_tokens},
        {"audio_ctx", params.audio_ctx},
        {"beam_size", params.beam_size},
        {"vad_thold", params.vad_thold},
        {"freq_thold", params.freq_thold},
        {"translate", params.translate},
        {"no_fallback", params.no_fallback},
        {"print_special", params.print_special},
        {"no_context", params.no_context},
        {"no_timestamps", params.no_timestamps},
        {"tinydiarize", params.tinydiarize},
        {"save_audio", params.save_audio},
        {"use_gpu", params.use_gpu},
        {"flash_attn", params.flash_attn},
        {"json_output", params.json_output},
        {"print_tokens", params.print_tokens},
        {"replay", params.replay},
        {"replay_file", params.replay_file},
        {"language", params.language},
        {"model", params.model},
        {"fname_out", params.fname_out}
    };
}

// Logger class to handle different output formats
class Logger {
private:
    bool json_mode;
    std::ofstream file_out;

public:
    Logger(const whisper_params& params) :
        json_mode(params.json_output) {

        // Open output file if specified
        if (!params.fname_out.empty()) {
            // In JSON mode, we write JSONL format
            file_out.open(params.fname_out);
            if (!file_out.is_open()) {
                fprintf(stderr, "Error: Failed to open output file '%s'\n", params.fname_out.c_str());
            }
        }
    }

    ~Logger() {
        if (file_out.is_open()) file_out.close();
    }

    // Log standard output (plain text or JSON based on mode)
    void log(const std::string& text, bool new_line = true) {
        if (json_mode) {
            json j = {
                {"type", "stdout"},
                {"text", text}
            };
            log_json(j);
        } else {
            printf("%s%s", text.c_str(), new_line ? "\n" : "");
            fflush(stdout);

            if (file_out.is_open()) {
                file_out << text << (new_line ? "\n" : "");
                file_out.flush();
            }
        }
    }

    // Log error message
    void error(const std::string& text) {
        if (json_mode) {
            json j = {
                {"type", "stderr"},
                {"text", text}
            };
            log_json(j);
        } else {
            fprintf(stderr, "Error: %s\n", text.c_str());

            if (file_out.is_open()) {
                file_out << "Error: " << text << std::endl;
            }
        }
    }

    // Log warning message
    void warning(const std::string& text) {
        if (json_mode) {
            json j = {
                {"type", "stderr"},
                {"text", "WARNING: " + text}
            };
            log_json(j);
        } else {
            fprintf(stderr, "WARNING: %s\n", text.c_str());

            if (file_out.is_open()) {
                file_out << "WARNING: " << text << std::endl;
            }
        }
    }

    // Log JSON directly
    void log_json(const json& j) {
        std::cout << j.dump(2) << std::endl << std::flush;

        // Write to file if open (always as JSONL format)
        if (file_out.is_open()) {
            file_out << j.dump() << std::endl;
            file_out.flush();
        }
    }

    // Check if we're in JSON output mode
    bool is_json_mode() const {
        return json_mode;
    }
};

std::vector<token_prob> get_token_probs(struct whisper_context* ctx, int segment_idx) {
    std::vector<token_prob> result;
    const int n_tokens = whisper_full_n_tokens(ctx, segment_idx);

    for (int i = 0; i < n_tokens; ++i) {
        const char* text = whisper_full_get_token_text(ctx, segment_idx, i);
        const float p = whisper_full_get_token_p(ctx, segment_idx, i);

        // Skip special tokens
        const whisper_token id = whisper_full_get_token_id(ctx, segment_idx, i);
        if (id >= whisper_token_eot(ctx)) {
            continue;
        }

        result.push_back({std::string(text), p});
    }

    return result;
}

void whisper_print_usage(int argc, char** argv, const whisper_params& params);

// Parse command line arguments
bool whisper_params_parse(int argc, char** argv, whisper_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-j"    || arg == "--json")          { params.json_output   = true; }
        else if (arg == "-pt"   || arg == "--print-tokens")  { params.print_tokens  = true; }
        else if (arg == "-r"    || arg == "--replay")        {
            params.replay      = true;
            params.replay_file = argv[++i];
        }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char** argv, const whisper_params& params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -j,       --json          [%-7s] output in JSON format\n",                          params.json_output ? "true" : "false");
    fprintf(stderr, "  -pt,      --print-tokens  [%-7s] include tokens in output\n",                       params.print_tokens ? "true" : "false");
    fprintf(stderr, "  -r FILE,  --replay FILE   [%-7s] replay transcriptions from JSONL file\n",          params.replay ? params.replay_file.c_str() : "false");
    fprintf(stderr, "\n");
}

// Forward declaration for parameters JSON output
void output_params_json(const whisper_params& params, Logger& logger);

// Function to output JSON for predictions and transcriptions
void generate_transcription_json(bool is_prediction, int iter, struct whisper_context* ctx,
                               Logger& logger, bool print_tokens, const int64_t iter_start_ms = 0) {
    // Create segments array
    json segments = json::array();
    const int n_segments = whisper_full_n_segments(ctx);

    std::string full_text;

    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        const bool speaker_turn = whisper_full_get_segment_speaker_turn_next(ctx, i);

        full_text += text;

        // Create a segment object with timestamps relative to iteration start
        json segment = {
            {"id", i},
            {"text", text},
            {"start_ms", iter_start_ms + (t0 * 10)},
            {"end_ms", iter_start_ms + (t1 * 10)},
            {"speaker_turn", speaker_turn}
        };

        // Always calculate average confidence
        float avg_confidence = 0.0f;
        const int n_tokens = whisper_full_n_tokens(ctx, i);
        int valid_tokens = 0;

        // Add token-level information if requested
        if (print_tokens) {
            json tokens = json::array();

            for (int j = 0; j < n_tokens; ++j) {
                const char* token_text = whisper_full_get_token_text(ctx, i, j);
                const float token_p = whisper_full_get_token_p(ctx, i, j);
                const whisper_token token_id = whisper_full_get_token_id(ctx, i, j);

                // Skip special tokens if we don't want to include them
                if (token_id >= whisper_token_eot(ctx)) {
                    continue;
                }

                tokens.push_back({
                    {"text", token_text},
                    {"p", token_p}
                });

                avg_confidence += token_p;
                valid_tokens++;
            }

            // Add tokens array to segment if we have any
            if (!tokens.empty()) {
                segment["tokens"] = tokens;
            }
        } else {
            // Just calculate confidence without storing tokens
            for (int j = 0; j < n_tokens; ++j) {
                const whisper_token token_id = whisper_full_get_token_id(ctx, i, j);
                if (token_id >= whisper_token_eot(ctx)) {
                    continue;
                }

                avg_confidence += whisper_full_get_token_p(ctx, i, j);
                valid_tokens++;
            }
        }

        // Add average confidence if we have tokens
        if (valid_tokens > 0) {
            avg_confidence /= valid_tokens;
            segment["confidence"] = avg_confidence;
        }

        segments.push_back(segment);
    }

    // Create the main JSON object
    json j = {
        {"type", is_prediction ? "prediction" : "transcription"},
        {"iter", iter},
        {"iter_start_ms", iter_start_ms},
        {"segments", segments},
        {"text", full_text}
    };

    // Output the JSON
    logger.log_json(j);
}

// Function to replay transcriptions from JSONL file
bool replay_transcriptions(const std::string& replay_file, Logger& logger) {
std::ifstream file(replay_file);
if (!file.is_open()) {
    logger.error("Failed to open replay file: " + replay_file);
    return false;
}

// Track the earliest timestamp for time adjustment
int64_t first_timestamp = -1;
int64_t replay_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

// Store all transcriptions with timestamps for ordered replay
struct ReplayItem {
    json data;
    int64_t timestamp;
    int iter;
};
std::vector<ReplayItem> replay_items;

// Read and parse the JSONL file
std::string line;
while (std::getline(file, line)) {
    try {
        json entry = json::parse(line);

        // Only process prediction/transcription entries with iter fields
        if (entry.contains("type") &&
            (entry["type"] == "prediction" || entry["type"] == "transcription") &&
            entry.contains("iter") &&
            entry.contains("iter_start_ms")) {

            int64_t timestamp = entry["iter_start_ms"];
            int iter = entry["iter"];

            // Track the earliest timestamp
            if (first_timestamp < 0 || timestamp < first_timestamp) {
                first_timestamp = timestamp;
            }

            replay_items.push_back({entry, timestamp, iter});
        }
    } catch (const std::exception& e) {
        logger.warning("Error parsing JSONL line: " + std::string(e.what()));
        // Continue with next line
    }
}

// Sort items by timestamp
std::sort(replay_items.begin(), replay_items.end(),
          [](const ReplayItem& a, const ReplayItem& b) {
              return a.timestamp < b.timestamp;
          });

logger.log("Replaying " + std::to_string(replay_items.size()) + " transcription items...");

// Replay the transcriptions with timing based on original intervals
for (const auto& item : replay_items) {
    // Calculate relative delay and add to replay start time
    int64_t relative_delay = item.timestamp - first_timestamp;
    int64_t target_time = replay_start_time + relative_delay;

    // Current time
    int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Sleep until the target time
    if (current_time < target_time) {
        std::this_thread::sleep_for(std::chrono::milliseconds(target_time - current_time));
    }

    // Output the transcription
    logger.log_json(item.data);
}

return true;
}

// Generate output for devices and model
void output_system_info(whisper_context* ctx, Logger& logger, const whisper_params& params) {
    if (!logger.is_json_mode()) return;

    // List audio devices
    json devices = json::array();
    int device_count = SDL_GetNumAudioDevices(SDL_TRUE); // Get capture (input) devices

    for (int i = 0; i < device_count; ++i) {
        const char* device_name = SDL_GetAudioDeviceName(i, SDL_TRUE);
        devices.push_back({
            {"id", i},
            {"name", device_name ? device_name : "Unknown Device " + std::to_string(i)}
        });
    }

    // Add default device
    devices.push_back({
        {"id", -1},
        {"name", "Default Device"}
    });

    // Create model info object
    json model_info = {
        {"name", ctx ? "Loaded" : "None"},
        {"type", ctx ? whisper_model_type_readable(ctx) : "Unknown"},
        {"multilingual", ctx ? whisper_is_multilingual(ctx) : false},
        {"vocab_size", ctx ? whisper_model_n_vocab(ctx) : 0},
        {"audio_ctx", ctx ? whisper_model_n_audio_ctx(ctx) : 0},
        {"text_ctx", ctx ? whisper_model_n_text_ctx(ctx) : 0}
    };

    // Create the main initialization JSON object (including params)
    json j = {
        {"type", "init"},
        {"devices", devices},
        {"model", model_info},
        {"params", create_params_json(params)}
    };

    // Output to logger
    logger.log_json(j);
}

// Output processing information
void output_processing_info(const whisper_params& params, Logger& logger,
                          bool use_vad, int n_samples_step, int n_samples_len,
                          int n_samples_keep, int n_new_line) {
    if (logger.is_json_mode()) {
        // Create processing metadata
        json j = {
            {"type", "processing-meta"},
            {"samples", {
                {"step", n_samples_step},
                {"step_sec", float(n_samples_step)/WHISPER_SAMPLE_RATE},
                {"length_sec", float(n_samples_len)/WHISPER_SAMPLE_RATE},
                {"keep_sec", float(n_samples_keep)/WHISPER_SAMPLE_RATE}
            }},
            {"threads", params.n_threads},
            {"language", params.language},
            {"task", params.translate ? "translate" : "transcribe"},
            {"timestamps", !params.no_timestamps},
            {"vad", use_vad}
        };

        if (!use_vad) {
            j["n_new_line"] = n_new_line;
            j["no_context"] = params.no_context;
        }

        // Output as JSON
        logger.log_json(j);
    } else {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len)/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }
}

// Process audio and generate transcript
bool process_audio(whisper_context* ctx, const whisper_params& params,
                  const std::vector<float>& pcmf32, Logger& logger,
                  int n_iter, bool use_vad, bool is_prediction,
                  const int64_t t_start_ms, std::vector<whisper_token>& prompt_tokens) {
    whisper_full_params wparams = whisper_full_default_params(
        params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.single_segment   = !use_vad;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.beam_search.beam_size = params.beam_size;
    wparams.audio_ctx        = params.audio_ctx;
    wparams.tdrz_enable      = params.tinydiarize;
    wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
    wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
    wparams.prompt_n_tokens  = params.no_context ? 0 : prompt_tokens.size();

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        logger.error("Failed to process audio");
        return false;
    }

    // Output results
    if (logger.is_json_mode()) {
        // Use elapsed time based on iteration number for more logical timing
        const int64_t elapsed_ms = t_start_ms;

        generate_transcription_json(is_prediction, n_iter, ctx, logger, params.print_tokens, elapsed_ms);
    } else {
        if (!use_vad) {
            printf("\33[2K\r");
            // print long empty line to clear the previous line
            printf("%s", std::string(100, ' ').c_str());
            printf("\33[2K\r");
        } else {
            const int64_t t1 = t_start_ms;
            const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

            printf("\n");
            printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
            printf("\n");
        }

        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char* text = whisper_full_get_segment_text(ctx, i);

            if (params.no_timestamps) {
                printf("%s", text);
                fflush(stdout);
            } else {
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                    output += " [SPEAKER_TURN]";
                }

                output += "\n";
                printf("%s", output.c_str());
                fflush(stdout);
            }
        }

        if (use_vad) {
            printf("\n");
            printf("### Transcription %d END\n", n_iter);
        }
    }

    return true;
}

// Handle audio buffer overflow
void handle_overflow(Logger& logger, audio_async& audio) {
    logger.warning("Cannot process audio fast enough, dropping audio...");
    audio.clear();
}



// Output parameters as standalone JSON (for backward compatibility)
void output_params_json(const whisper_params& params, Logger& logger) {
    json j = {
        {"type", "params"},
        {"params", create_params_json(params)}
    };

    logger.log_json(j);
}
int main(int argc, char** argv) {
    whisper_params params;

    if (!whisper_params_parse(argc, argv, params)) {
        return 1;
    }

    // Initialize the logger
    Logger logger(params);

    // Check if we're in replay mode
    if (params.replay) {
        // Force JSON output in replay mode
        params.json_output = true;

        logger.log("Starting replay from file: " + params.replay_file);

        // Replay transcriptions from file and exit
        if (!replay_transcriptions(params.replay_file, logger)) {
            return 1;
        }

        logger.log("Replay completed successfully");
        return 0;
    }

    // Normal transcription mode - continue with initialization
    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD
    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // Initialize audio
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        logger.error("Failed to initialize audio");
        return 1;
    }

    audio.resume();

    // Initialize Whisper
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        logger.error("Unknown language: " + params.language);
        whisper_print_usage(argc, argv, params);
        return 1;
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (!ctx) {
        logger.error("Failed to initialize whisper context");
        return 1;
    }

    // Output system information (including params)
    output_system_info(ctx, logger, params);

    // Check model compatibility
    if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
            logger.warning("Model is not multilingual, ignoring language and translation options");
            params.language = "en";
            params.translate = false;
        }
    }

    // Output processing information
    output_processing_info(params, logger, use_vad, n_samples_step, n_samples_len, n_samples_keep, n_new_line);

    // Initialize audio buffers
    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    std::vector<whisper_token> prompt_tokens;

    // Initialize WAV writer for saving audio if requested
    wav_writer wavWriter;
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }

    // Let the user know we're ready
    logger.log("[Start speaking]");

    // Main processing loop
    int n_iter = 0;
    bool is_running = true;
    auto t_last = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;
    int64_t iter_start_ms = 0;

    while (is_running) {
        // Save audio if requested
        if (params.save_audio && !pcmf32_new.empty()) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }

        // Handle Ctrl+C
        is_running = sdl_poll_events();
        if (!is_running) break;

        // Process audio based on mode (VAD or fixed step)
        if (!use_vad) {
            // Fixed step mode
            while (true) {
                is_running = sdl_poll_events();
                if (!is_running) break;

                audio.get(params.step_ms, pcmf32_new);

                // Handle overflow
                if ((int)pcmf32_new.size() > 2*n_samples_step) {
                    handle_overflow(logger, audio);
                    continue;
                }

                if ((int)pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // Take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int)pcmf32_old.size(),
                                             std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            // Prepare the audio buffer
            pcmf32.resize(n_samples_new + n_samples_take);

            // Copy old audio samples
            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            // Copy new audio samples
            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            // Save current audio for next iteration
            pcmf32_old = pcmf32;
        } else {
            // VAD mode
            const auto t_now = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(2000, pcmf32_new);

            // Check for voice activity
            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            t_last = t_now;
        }

        // Process the audio and generate transcript
        const bool is_prediction = !use_vad && (n_iter % n_new_line) != 0;

        // Always use the actual time difference from start
        iter_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t_start).count();

        if (!process_audio(ctx, params, pcmf32, logger, n_iter, use_vad,
                          is_prediction, iter_start_ms, prompt_tokens)) {
            break;
        }

        ++n_iter;

        // Handle end of audio segment (new line)
        if (!use_vad && (n_iter % n_new_line) == 0) {
            if (!logger.is_json_mode()) {
                printf("\n");
            }

            // Keep part of the audio for next iteration to mitigate word boundary issues
            pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

            // Add tokens of the last full length segment as the prompt
            if (!params.no_context) {
                prompt_tokens.clear();

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const int token_count = whisper_full_n_tokens(ctx, i);
                    for (int j = 0; j < token_count; ++j) {
                        prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                    }
                }
            }
        }
        fflush(stdout);
    }

    // Cleanup
    audio.pause();
    if (params.save_audio) {
        wavWriter.close();
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
