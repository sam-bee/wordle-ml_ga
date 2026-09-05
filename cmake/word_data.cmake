# Freeze the exact model-facing vocabulary order and training split. Embedding the
# validated bytes also lets the file loader reject substitutions after compilation
# without introducing a runtime cryptography dependency.
set(WORD_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(word_data_header "// Generated from the frozen word lists.\n#pragma once\nnamespace wordle_ga::fitness::frozen {\n")

macro(freeze_word_data filename expected_sha symbol)
    set(word_data_path "${WORD_DATA_DIR}/${filename}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${word_data_path}")
    file(SHA256 "${word_data_path}" actual_sha)
    if(NOT actual_sha STREQUAL "${expected_sha}")
        message(FATAL_ERROR "Frozen word data checksum mismatch: ${filename}")
    endif()
    file(READ "${word_data_path}" contents)
    string(APPEND word_data_header "inline constexpr char ${symbol}[] = R\"words(${contents})words\";\n")
endmacro()

freeze_word_data(wordlist-action-space-4739.csv
    992239bab3de16bf51dcca2bc10efe5f81c6d92e01f00172f96574518807f4eb kActions)
freeze_word_data(wordlist-valid-solutions-all-2309.csv
    66d2f19d4543833517c79aafa44a6582251c21618157bffcd9e453daf405d4ff kSolutions)
freeze_word_data(wordlist-valid-solutions-train-2109.csv
    70184dfa5c291a73c8576f8ea6bbe041482890b63b68e965e9c94069577f7b78 kTraining)
string(APPEND word_data_header "}\n")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/frozen_word_data.hpp" CONTENT "${word_data_header}" @ONLY)
