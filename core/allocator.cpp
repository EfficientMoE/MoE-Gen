// clang-format off
/* ----------------------------------------------------------------------------  *
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
// clang-format on

#include "allocator.h"

extern "C" {
void* TorchAllocate(size_t n) {
    // std::cerr << "Allocating data" << std::endl;
    return malloc(n);
}

void TorchFree(void* ptr) {
    // std::cerr << "Freeing data" << std::endl;
    free(ptr);
}
}

ReplaceTorchAllocatorOnLoad kReplaceTorchAllocatorOnLoad;
