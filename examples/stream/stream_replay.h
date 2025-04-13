#pragma once

#include <string>

// Forward declarations
class Logger;

/**
 * Replays transcriptions from a JSONL file.
 * 
 * This function reads a JSONL file containing transcription data
 * and replays them with appropriate timing between entries.
 * 
 * @param replay_file Path to the JSONL file containing transcription data
 * @param logger Logger instance to output replayed data and messages
 * @return True if replay was successful, false otherwise
 */
bool replay_transcriptions(const std::string& replay_file, Logger& logger);