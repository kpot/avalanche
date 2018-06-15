#ifndef AVALANCHE_LOGGING_H
#define AVALANCHE_LOGGING_H

#include <memory>

#include "spdlog/spdlog.h"

namespace avalanche {


std::shared_ptr<spdlog::logger> get_logger();

} // namespace

#endif //AVALANCHE_LOGGING_H
