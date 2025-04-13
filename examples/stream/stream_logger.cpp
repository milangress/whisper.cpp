#include "stream_logger.h"
#include <iostream>

Logger::Logger(const whisper_params& params) :
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

Logger::~Logger() {
    if (file_out.is_open()) file_out.close();
}

// Log standard output (plain text or JSON based on mode)
void Logger::log(const std::string& text, bool new_line) {
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
void Logger::error(const std::string& text) {
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
void Logger::warning(const std::string& text) {
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
void Logger::log_json(const json& j) {
    std::cout << j.dump(2) << std::endl << std::flush;

    // Write to file if open (always as JSONL format)
    if (file_out.is_open()) {
        file_out << j.dump() << std::endl;
        file_out.flush();
    }
}

// Check if we're in JSON output mode
bool Logger::is_json_mode() const {
    return json_mode;
}
