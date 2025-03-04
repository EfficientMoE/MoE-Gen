/* ---------------------------------------------------------------------------- *
 *  MoE-Gen                                                                      *
 *  copyright (c) EfficientMoE team 2025                                             *
 *                                                                               *
 *  licensed under the apache license, version 2.0 (the "license");              *
 *  you may not use this file except in compliance with the license.             *
 *                                                                               *
 *  you may obtain a copy of the license at                                      *
 *                                                                               *
 *                  http://www.apache.org/licenses/license-2.0                   *
 *                                                                               *
 *  unless required by applicable law or agreed to in writing, software          *
 *  distributed under the license is distributed on an "as is" basis,            *
 *  without warranties or conditions of any kind, either express or implied.     *
 *  see the license for the specific language governing permissions and          *
 *  limitations under the license.                                               *
 * ---------------------------------------------------------------------------- */

// Copyright(c) 2015-present, Gabi Melman & spdlog contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

#pragma once

#ifndef SPDLOG_HEADER_ONLY
    #include <spdlog/sinks/basic_file_sink.h>
#endif

#include <spdlog/common.h>
#include <spdlog/details/os.h>

namespace spdlog {
namespace sinks {

template <typename Mutex>
SPDLOG_INLINE basic_file_sink<Mutex>::basic_file_sink(const filename_t &filename,
                                                      bool truncate,
                                                      const file_event_handlers &event_handlers)
    : file_helper_{event_handlers} {
    file_helper_.open(filename, truncate);
}

template <typename Mutex>
SPDLOG_INLINE const filename_t &basic_file_sink<Mutex>::filename() const {
    return file_helper_.filename();
}

template <typename Mutex>
SPDLOG_INLINE void basic_file_sink<Mutex>::truncate() {
    std::lock_guard<Mutex> lock(base_sink<Mutex>::mutex_);
    file_helper_.reopen(true);
}

template <typename Mutex>
SPDLOG_INLINE void basic_file_sink<Mutex>::sink_it_(const details::log_msg &msg) {
    memory_buf_t formatted;
    base_sink<Mutex>::formatter_->format(msg, formatted);
    file_helper_.write(formatted);
}

template <typename Mutex>
SPDLOG_INLINE void basic_file_sink<Mutex>::flush_() {
    file_helper_.flush();
}

}  // namespace sinks
}  // namespace spdlog
