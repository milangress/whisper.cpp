#pragma once

#include "stream_common.h"
#include <fstream>

/**
 * Logger class to handle different output formats (text or JSON)
 */
class Logger {
private:
    bool json_mode;
    std::ofstream file_out;

public:
    /**
     * Constructor
     * @param params The whisper parameters with output settings
     */
    Logger(const whisper_params& params);

    /**
     * Destructor - closes any open file handles
     */
    ~Logger();

    /**
     * Log standard output (plain text or JSON based on mode)
     * @param text The text to log
     * @param new_line Whether to add a newline after the text
     */
    void log(const std::string& text, bool new_line = true);

    /**
     * Log error message
     * @param text The error message
     */
    void error(const std::string& text);

    /**
     * Log warning message
     * @param text The warning message
     */
    void warning(const std::string& text);

    /**
     * Log JSON directly
     * @param j The JSON object to log
     */
    void log_json(const json& j);

    /**
     * Check if we're in JSON output mode
     * @return true if in JSON mode, false otherwise
     */
    bool is_json_mode() const;
};
