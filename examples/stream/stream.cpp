// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

// Include the new utility header files
#include "stream_common.h"
#include "stream_logger.h"
#include "stream_replay.h"
#include "stream_utils.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

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
    
    // If get-audio-devices flag is set, exit after listing devices
    if (params.get_audio_devices) {
        whisper_free(ctx);
        return 0;
    }

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

        // Force flush after each transcription to ensure immediate delivery
        fflush(stdout);
        fflush(stderr);
        
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
