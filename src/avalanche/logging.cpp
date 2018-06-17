#include <mutex>

#include "spdlog/spdlog.h"

#include "avalanche/logging.h"

namespace avalanche {

static std::mutex logging_initializer_access;
static std::shared_ptr<spdlog::logger> default_console_logger;

std::shared_ptr<spdlog::logger> get_logger() {
    std::lock_guard<std::mutex> lock(logging_initializer_access);
    if (!default_console_logger) {
        default_console_logger = spdlog::stdout_color_mt("console");
        default_console_logger->set_level(spdlog::level::info);
    }
    return default_console_logger;
}

} // namespace
