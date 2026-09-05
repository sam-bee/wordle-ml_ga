#include "fitness/vocabulary.hpp"
#include "frozen_word_data.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace wordle_ga::fitness {
namespace {

Word ParseWord(const std::string &line, const std::string &path, std::size_t line_number) {
    std::string_view word(line);
    if (!word.empty() && word.back() == '\r')
        word.remove_suffix(1);
    if (word.size() != 5) {
        throw std::runtime_error(path + ": line " + std::to_string(line_number) + " is not a five-letter word");
    }
    Word result{};
    for (std::size_t i = 0; i < word.size(); ++i) {
        const unsigned char character = static_cast<unsigned char>(word[i]);
        if (character < 'A' || character > 'Z') {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) + " is not uppercase ASCII");
        }
        result.letters[i] = static_cast<std::uint8_t>(character - 'A');
    }
    return result;
}

template <std::size_t Count> std::array<Word, Count> ReadWords(const std::string &path, std::string_view frozen) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open vocabulary file: " + path);

    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (bytes != frozen)
        throw std::runtime_error("vocabulary file does not match frozen data: " + path);

    std::array<Word, Count> words{};
    std::istringstream lines(bytes);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(lines, line)) {
        ++line_number;
        if (line_number > Count)
            throw std::runtime_error(path + ": too many words");
        words[line_number - 1] = ParseWord(line, path, line_number);
    }
    if (line_number != Count) {
        throw std::runtime_error(path + ": expected " + std::to_string(Count) + " words, got " +
                                 std::to_string(line_number));
    }
    for (std::size_t i = 1; i < Count; ++i) {
        const auto &previous = words[i - 1];
        const auto &current = words[i];
        if (!std::lexicographical_compare(previous.letters, previous.letters + 5, current.letters,
                                          current.letters + 5)) {
            throw std::runtime_error(path + ": words are not strictly sorted and unique");
        }
    }
    return words;
}

std::string Path(const std::string &directory, const char *filename) {
    if (directory.empty())
        return filename;
    return directory.back() == '/' ? directory + filename : directory + "/" + filename;
}

} // namespace

Vocabulary LoadVocabulary(const std::string &data_directory) {
    const auto actions =
        ReadWords<model::kNumActions>(Path(data_directory, "wordlist-action-space-4739.csv"), frozen::kActions);
    const auto solutions = ReadWords<model::kNumSolutions>(
        Path(data_directory, "wordlist-valid-solutions-all-2309.csv"), frozen::kSolutions);
    const auto training = ReadWords<kTrainingSolutions>(Path(data_directory, "wordlist-valid-solutions-train-2109.csv"),
                                                        frozen::kTraining);
    Vocabulary vocabulary{actions, solutions, {}, {}};

    std::unordered_map<std::string, std::uint16_t> solution_ids;
    solution_ids.reserve(model::kNumSolutions);
    for (std::size_t i = 0; i < vocabulary.solutions.size(); ++i) {
        std::string key;
        key.reserve(5);
        for (std::uint8_t letter : vocabulary.solutions[i].letters)
            key.push_back(static_cast<char>('A' + letter));
        solution_ids.emplace(std::move(key), static_cast<std::uint16_t>(i));
    }
    // Action and solution IDs are independent; map by word identity.
    for (std::size_t i = 0; i < model::kNumSolutions; ++i) {
        const auto action = std::find_if(vocabulary.actions.begin(), vocabulary.actions.end(), [&](const Word &word) {
            for (int letter = 0; letter < 5; ++letter) {
                if (word.letters[letter] != vocabulary.solutions[i].letters[letter])
                    return false;
            }
            return true;
        });
        if (action == vocabulary.actions.end())
            throw std::runtime_error("solution is missing from action vocabulary");
        vocabulary.solution_actions[i] = static_cast<std::uint16_t>(action - vocabulary.actions.begin());
    }
    for (std::size_t i = 0; i < training.size(); ++i) {
        std::string key;
        for (std::uint8_t letter : training[i].letters)
            key.push_back(static_cast<char>('A' + letter));
        const auto found = solution_ids.find(key);
        if (found == solution_ids.end())
            throw std::runtime_error("training word is missing from all-solutions vocabulary");
        vocabulary.training_solutions[i] = found->second;
    }
    return vocabulary;
}

} // namespace wordle_ga::fitness
