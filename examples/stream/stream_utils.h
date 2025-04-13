#pragma once

#include "stream_common.h"
#include "common-sdl.h"
#include "whisper.h"

/**
 * Get token probabilities for a segment
 * @param ctx The whisper context
 * @param segment_idx The segment index to get tokens from
 * @return Vector of token probability structures
 */
std::vector<token_prob> get_token_probs(struct whisper_context* ctx, int segment_idx);

/**
 * Handle audio buffer overflow
 * @param logger The logger instance
 * @param audio The audio instance to clear
 */
void handle_overflow(Logger& logger, audio_async& audio);

/**
 * Generate JSON for parameter configuration
 * @param params The parameters to convert to JSON
 * @return JSON object containing the parameters
 */
json create_params_json(const whisper_params& params);

/**
 * Output system information as JSON
 * @param ctx The whisper context
 * @param logger The logger instance
 * @param params The whisper parameters
 */
void output_system_info(whisper_context* ctx, Logger& logger, const whisper_params& params);

/**
 * Output processing configuration information
 * @param params The whisper parameters
 * @param logger The logger instance
 * @param use_vad Whether VAD is enabled
 * @param n_samples_step Number of samples per step
 * @param n_samples_len Length of samples
 * @param n_samples_keep Number of samples to keep
 * @param n_new_line Number of iterations before new line
 */
void output_processing_info(const whisper_params& params, Logger& logger,
                            bool use_vad, int n_samples_step, int n_samples_len,
                            int n_samples_keep, int n_new_line);

/**
 * Generate JSON for transcription results
 * @param is_prediction Whether this is a prediction (vs. final transcription)
 * @param iter The iteration number
 * @param ctx The whisper context
 * @param logger The logger instance
 * @param print_tokens Whether to include token probabilities
 * @param iter_start_ms The timestamp of iteration start
 */
void generate_transcription_json(bool is_prediction, int iter, struct whisper_context* ctx,
                                Logger& logger, bool print_tokens, const int64_t iter_start_ms = 0);

/**
 * Output parameters as standalone JSON
 * @param params The whisper parameters
 * @param logger The logger instance
 */
void output_params_json(const whisper_params& params, Logger& logger);

/**
 * Process audio and generate transcript
 * @param ctx The whisper context
 * @param params The whisper parameters
 * @param pcmf32 The audio samples
 * @param logger The logger instance
 * @param n_iter The iteration number
 * @param use_vad Whether VAD is enabled
 * @param is_prediction Whether this is a prediction
 * @param t_start_ms The start timestamp
 * @param prompt_tokens The prompt tokens
 * @return Whether processing was successful
 */
bool process_audio(whisper_context* ctx, const whisper_params& params,
                  const std::vector<float>& pcmf32, Logger& logger,
                  int n_iter, bool use_vad, bool is_prediction,
                  const int64_t t_start_ms, std::vector<whisper_token>& prompt_tokens);
