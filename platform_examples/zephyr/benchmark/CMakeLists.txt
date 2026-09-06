cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(emlearn_benchmark)

# Find emlearn headers using Python
# This works both when building in-place and when app is copied to temp directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import emlearn; print(emlearn.includedir)"
    OUTPUT_VARIABLE EMLEARN_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE EMLEARN_RESULT
)
if(NOT EMLEARN_RESULT EQUAL 0 OR NOT EXISTS "${EMLEARN_INCLUDE_DIR}")
    # Fallback: try relative path for in-place builds
    set(EMLEARN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../emlearn")
endif()
message(STATUS "emlearn include dir: ${EMLEARN_INCLUDE_DIR}")

target_include_directories(app PRIVATE ${EMLEARN_INCLUDE_DIR})

file(GLOB app_sources src/*.c)
target_sources(app PRIVATE ${app_sources})

# CMSIS-DSP optimization (for tree voting vectorization)
if(CONFIG_EMLEARN_USE_CMSIS_DSP)
    target_compile_definitions(app PRIVATE EML_TREES_CMSIS_DSP=1)
endif()

# Pointer-based tree traversal (alternative to index-based)
if(CONFIG_EMLEARN_POINTER_TRAVERSAL)
    target_compile_definitions(app PRIVATE EML_TREES_USE_POINTER_TRAVERSAL=1)
endif()
