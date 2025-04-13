#include "stream_replay.h"
#include "stream_logger.h"
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>

bool replay_transcriptions(const std::string& replay_file, Logger& logger) {
    std::ifstream file(replay_file);
    if (!file.is_open()) {
        logger.error("Failed to open replay file: " + replay_file);
        return false;
    }

    // Store transcriptions in the order they appear (should match iter order)
    std::vector<json> transcriptions;

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

                transcriptions.push_back(entry);
            }
        } catch (const std::exception& e) {
            logger.warning("Error parsing JSONL line: " + std::string(e.what()));
            // Continue with next line
        }
    }

    if (transcriptions.empty()) {
        logger.warning("No transcription items found in replay file");
        return false;
    }

    logger.log("Replaying " + std::to_string(transcriptions.size()) + " transcription items...");

    // Determine the earliest timestamp to calculate relative delays
    int64_t first_timestamp = transcriptions[0]["iter_start_ms"];
    for (const auto& entry : transcriptions) {
        if (entry["iter_start_ms"] < first_timestamp) {
            first_timestamp = entry["iter_start_ms"];
        }
    }

    // Start time of replay
    int64_t replay_start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Previous target time, used to handle out-of-order timestamps
    int64_t prev_target_time = replay_start_time;

    // Replay the transcriptions in the order they appear in the file
    for (const auto& entry : transcriptions) {
        int64_t timestamp = entry["iter_start_ms"];

        // Calculate when this item should be shown
        int64_t relative_delay = timestamp - first_timestamp;
        int64_t target_time = replay_start_time + relative_delay;

        // If target time is before previous, show immediately
        // Otherwise sleep until the target time
        if (target_time > prev_target_time) {
            int64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();

            if (current_time < target_time) {
                std::this_thread::sleep_for(std::chrono::milliseconds(target_time - current_time));
            }
            prev_target_time = target_time;
        }

        // Output the transcription
        logger.log_json(entry);
    }

    return true;
}
