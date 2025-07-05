project(AudioExample)

add_executable(audio_example Source/Main.cpp)

target_link_libraries(audio_example PRIVATE rtpghi)

target_include_directories(audio_example
    PRIVATE
    ${dj_fft_SOURCE_DIR}
    ${dr_wav_SOURCE_DIR}
)

set_project_warnings(audio_example)

# Suppress dr_wav header warnings (cast-align, old-style-cast, sign-conversion, string-conversion)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(audio_example PRIVATE
        -Wno-old-style-cast
        -Wno-implicit-int-conversion
        -Wno-cast-align
        -Wno-sign-conversion
        -Wno-string-conversion
        -Wno-implicit-float-conversion
        -Wno-double-promotion
    )
endif()
