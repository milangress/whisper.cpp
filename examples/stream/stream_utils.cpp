#include "stream_utils.h"
#include "common-whisper.h"
#include "stream_logger.h"
#include "stream_common.h"

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
        {"fname_out", params.fname_out},
        {"get_audio_devices", params.get_audio_devices}
    };
}

// Helper function to get token probabilities for a segment
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
        else if (arg == "-gad"  || arg == "--get-audio-devices") { params.get_audio_devices = true; }
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
    fprintf(stderr, "  -gad,     --get-audio-devices [%-7s] just list audio devices and exit\n",           params.get_audio_devices ? "true" : "false");
    fprintf(stderr, "\n");
}

// Handle audio buffer overflow
void handle_overflow(Logger& logger, audio_async& audio) {
    logger.warning("Cannot process audio fast enough, dropping audio...");
    audio.clear();
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

// Function to output JSON for predictions and transcriptions
void generate_transcription_json(bool is_prediction, int iter, struct whisper_context* ctx,
                               Logger& logger, bool print_tokens, const int64_t iter_start_ms) {
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
    
    // Additional explicit flush to ensure immediate delivery
    fflush(stdout);
}

// Output parameters as standalone JSON (for backward compatibility)
void output_params_json(const whisper_params& params, Logger& logger) {
    json j = {
        {"type", "params"},
        {"params", create_params_json(params)}
    };

    logger.log_json(j);
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
