if (WHISPER_SDL2)
    set(TARGET whisper-stream)
add_executable(${TARGET}
    stream.cpp
    stream_utils.cpp
    stream_logger.cpp
    stream_replay.cpp
    )

    include(DefaultTargetOptions)

# Define version macros passed from the main CMakeLists.txt
# These macros will be available in the source files compiled for ${TARGET}
target_compile_definitions(${TARGET} PRIVATE
    WHISPER_CPP_VERSION="${PROJECT_VERSION}"
    PLACE_MILAN_WHISPER_VERSION="${PLACE_MILAN_WHISPER_VERSION}"
)
    target_link_libraries(${TARGET} PRIVATE common common-sdl whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME)
endif ()
