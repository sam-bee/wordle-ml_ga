#include "fitness/game.cuh"
#include "fitness/vocabulary.hpp"
#include "genotype_slab/device.cuh"
#include "genotype_slab/slab.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "golden_cases.hpp"

namespace f = wordle_ga::fitness;
namespace m = wordle_ga::model;
namespace gs = wordle_ga::genotype_slab;
cudaStream_t gStream = nullptr;

namespace {
void Require(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}
std::vector<float> Floats(const char *name, std::size_t count) {
    std::ifstream file(std::string(POLICY_FIXTURE_DIR) + "/" + name, std::ios::binary | std::ios::ate);
    Require(file && file.tellg() == static_cast<std::streamoff>(count * sizeof(float)), "bad golden fixture");
    std::vector<float> out(count);
    file.seekg(0);
    Require(static_cast<bool>(file.read(reinterpret_cast<char *>(out.data()), out.size() * 4)), "golden read");
    return out;
}
int Action(const f::Vocabulary &v, const char *word) {
    f::Word w{};
    for (int i = 0; i < 5; ++i)
        w.letters[i] = static_cast<std::uint8_t>(word[i] - 'A');
    auto it = std::find_if(v.actions.begin(), v.actions.end(),
                           [&](const f::Word &x) { return std::memcmp(&x, &w, sizeof(w)) == 0; });
    Require(it != v.actions.end(), "word absent from actions");
    return int(it - v.actions.begin());
}
int Solution(const f::Vocabulary &v, const char *word) {
    f::Word w{};
    for (int i = 0; i < 5; ++i)
        w.letters[i] = static_cast<std::uint8_t>(word[i] - 'A');
    auto it = std::find_if(v.solutions.begin(), v.solutions.end(),
                           [&](const f::Word &x) { return std::memcmp(&x, &w, sizeof(w)) == 0; });
    Require(it != v.solutions.end(), "word absent from solutions");
    return int(it - v.solutions.begin());
}
void Copy(const void *src, void *dst, std::size_t bytes, cudaMemcpyKind kind) {
    CheckCuda(cudaMemcpyAsync(dst, src, bytes, kind, gStream), "cudaMemcpyAsync");
}

__global__ void AllocateOne(gs::DeviceView slab, gs::Slot *out) {
    if (threadIdx.x == 0)
        out[0] = gs::Allocate(slab);
}

void EncodeGolden(const f::Vocabulary &v) {
    const auto values = Floats("golden-vectors.f32le", kGoldenValueCount);
    constexpr int n = static_cast<int>(std::size(kGoldenCases));
    std::array<m::Input, n> host{};
    std::array<f::Game, n> games{};
    std::array<int, n> active{};
    for (int i = 0; i < n; ++i) {
        std::copy_n(values.data() + kGoldenCases[i].candidates, m::kNumSolutions, host[i].candidate_mask);
        std::copy_n(values.data() + kGoldenCases[i].stats, m::kCandidateStatsSize, host[i].candidate_stats);
        std::copy_n(values.data() + kGoldenCases[i].remaining, m::kNumActions, host[i].remaining_action_mask);
        games[i].guesses = kGoldenCases[i].turn;
        active[i] = 1;
        for (int s = 0; s < m::kNumSolutions; ++s)
            games[i].candidates[s] = host[i].candidate_mask[s] != 0;
    }
    f::Word *sol = nullptr;
    std::uint16_t *map = nullptr;
    f::Game *dg = nullptr;
    int *da = nullptr;
    m::Input *di = nullptr;
    CheckCuda(cudaMalloc(&sol, sizeof(v.solutions)), "solutions");
    CheckCuda(cudaMalloc(&map, sizeof(v.solution_actions)), "map");
    CheckCuda(cudaMalloc(&dg, sizeof(games)), "games");
    CheckCuda(cudaMalloc(&da, sizeof(active)), "active");
    CheckCuda(cudaMalloc(&di, sizeof(host)), "inputs");
    Copy(v.solutions.data(), sol, sizeof(v.solutions), cudaMemcpyHostToDevice);
    Copy(v.solution_actions.data(), map, sizeof(v.solution_actions), cudaMemcpyHostToDevice);
    Copy(games.data(), dg, sizeof(games), cudaMemcpyHostToDevice);
    Copy(active.data(), da, sizeof(active), cudaMemcpyHostToDevice);
    f::Tables t{};
    t.solutions = sol;
    t.solution_actions = map;
    CheckCuda(f::EncodeGames(t, dg, da, n, di, gStream), "EncodeGames");
    CheckCuda(cudaStreamSynchronize(gStream), "Encode sync");
    Copy(di, host.data(), sizeof(host), cudaMemcpyDeviceToHost);
    CheckCuda(cudaStreamSynchronize(gStream), "golden result copy");
    for (int i = 0; i < n; ++i) {
        Require(games[i].guesses == kGoldenCases[i].turn, "golden turn mismatch");
        for (int j = 0; j < m::kNumSolutions; ++j)
            Require(host[i].candidate_mask[j] == (games[i].candidates[j] ? 1.0f : 0.0f), "candidate mask mismatch");
        for (int j = 0; j < m::kCandidateStatsSize; ++j)
            Require(std::fabs(host[i].candidate_stats[j] - values[kGoldenCases[i].stats + j]) <= 1e-6f,
                    "stats mismatch");
        for (int j = 0; j < m::kNumActions; ++j)
            Require(host[i].remaining_action_mask[j] == values[kGoldenCases[i].remaining + j],
                    "remaining mask mismatch");
    }
    cudaFree(di);
    cudaFree(da);
    cudaFree(dg);
    cudaFree(map);
    cudaFree(sol);
}

void AdvanceTests(const f::Vocabulary &v) {
    f::Word *a = nullptr, *s = nullptr;
    std::uint8_t *feedback = nullptr;
    f::Game *g = nullptr;
    int *active = nullptr;
    float *logits = nullptr;
    const std::size_t fc = std::size_t(m::kNumActions) * m::kNumSolutions;
    CheckCuda(cudaMalloc(&a, sizeof(v.actions)), "actions");
    CheckCuda(cudaMalloc(&s, sizeof(v.solutions)), "solutions");
    CheckCuda(cudaMalloc(&feedback, fc), "feedback");
    CheckCuda(cudaMalloc(&g, sizeof(f::Game)), "game");
    CheckCuda(cudaMalloc(&active, sizeof(int)), "active");
    CheckCuda(cudaMalloc(&logits, m::kNumActions * sizeof(float)), "logits");
    Copy(v.actions.data(), a, sizeof(v.actions), cudaMemcpyHostToDevice);
    Copy(v.solutions.data(), s, sizeof(v.solutions), cudaMemcpyHostToDevice);
    CheckCuda(f::BuildFeedback(a, s, feedback, gStream), "BuildFeedback");
    f::Tables t{a, s, nullptr, nullptr, feedback};
    std::vector<float> row(m::kNumActions, 0.0f);
    auto run = [&](f::Game game, int on, int x, int y) {
        game.target = Solution(v, "ABACK");
        Copy(&game, g, sizeof(game), cudaMemcpyHostToDevice);
        Copy(&on, active, 4, cudaMemcpyHostToDevice);
        std::fill(row.begin(), row.end(), 0);
        row[x] = 1;
        if (y >= 0)
            row[y] = 1;
        Copy(row.data(), logits, row.size() * 4, cudaMemcpyHostToDevice);
        CheckCuda(f::AdvanceGames(t, g, active, 1, logits, gStream), "AdvanceGames");
        CheckCuda(cudaStreamSynchronize(gStream), "Advance sync");
        Copy(g, &game, sizeof(game), cudaMemcpyDeviceToHost);
        Copy(active, &on, 4, cudaMemcpyDeviceToHost);
        CheckCuda(cudaStreamSynchronize(gStream), "result copy sync");
        return std::pair<f::Game, int>{game, on};
    };
    f::Game base{};
    std::fill(std::begin(base.candidates), std::end(base.candidates), 1);
    base.guesses = 0;
    const int low = Action(v, "ABACK"), high = Action(v, "ABASE");
    const int probe = Action(v, "AARGH");
    auto r = run(base, 1, probe, -1);
    Require(r.second == 1 && !r.first.won && r.first.history[0] == probe, "probe selection failed");
    r = run(base, 1, low, -1);
    Require(r.second == 0 && r.first.won && r.first.history[0] == low, "correct guess win failed");
    r = run(base, 1, low, high);
    Require(r.first.history[0] == std::min(low, high), "low-ID tie selection failed");
    base = {};
    std::fill(std::begin(base.candidates), std::end(base.candidates), 1);
    base.guesses = 1;
    base.history[0] = low;
    r = run(base, 1, low, high);
    Require(r.second == 1 && !r.first.won && r.first.history[1] == high, "repeat suppression failed");
    base = {};
    std::fill(std::begin(base.candidates), std::end(base.candidates), 1);
    base.guesses = 5;
    base.history[0] = Action(v, "ARISE");
    r = run(base, 1, high, -1);
    Require(r.second == 0 && r.first.guesses == 6 && !r.first.won, "sixth loss failed");
    base = {};
    std::fill(std::begin(base.candidates), std::end(base.candidates), 1);
    base.guesses = 0;
    row.assign(m::kNumActions, 0);
    row[low] = NAN;
    Copy(&base, g, sizeof(base), cudaMemcpyHostToDevice);
    int one = 1;
    Copy(&one, active, 4, cudaMemcpyHostToDevice);
    Copy(row.data(), logits, row.size() * 4, cudaMemcpyHostToDevice);
    CheckCuda(f::AdvanceGames(t, g, active, 1, logits, gStream), "NaN advance");
    CheckCuda(cudaStreamSynchronize(gStream), "NaN sync");
    Copy(g, &base, sizeof(base), cudaMemcpyDeviceToHost);
    Require(base.invalid == 1, "nonfinite invalid failed");
    base = {};
    row.assign(m::kNumActions, 0);
    row[low] = INFINITY;
    Copy(&base, g, sizeof(base), cudaMemcpyHostToDevice);
    Copy(&one, active, 4, cudaMemcpyHostToDevice);
    Copy(row.data(), logits, row.size() * 4, cudaMemcpyHostToDevice);
    CheckCuda(f::AdvanceGames(t, g, active, 1, logits, gStream), "Inf advance");
    CheckCuda(cudaStreamSynchronize(gStream), "Inf sync");
    Copy(g, &base, sizeof(base), cudaMemcpyDeviceToHost);
    CheckCuda(cudaStreamSynchronize(gStream), "Inf copy");
    Require(base.invalid == 1, "infinite invalid failed");
    base = {};
    std::memset(&base, 0x5a, sizeof(base));
    int zero = 0;
    row.assign(m::kNumActions, NAN);
    Copy(&base, g, sizeof(base), cudaMemcpyHostToDevice);
    Copy(&zero, active, 4, cudaMemcpyHostToDevice);
    Copy(row.data(), logits, row.size() * 4, cudaMemcpyHostToDevice);
    CheckCuda(f::AdvanceGames(t, g, active, 1, logits, gStream), "finished advance");
    CheckCuda(cudaStreamSynchronize(gStream), "finished sync");
    f::Game untouched{};
    int untouched_active = -1;
    Copy(g, &untouched, sizeof(untouched), cudaMemcpyDeviceToHost);
    Copy(active, &untouched_active, 4, cudaMemcpyDeviceToHost);
    CheckCuda(cudaStreamSynchronize(gStream), "finished copy");
    Require(std::memcmp(&base, &untouched, sizeof(base)) == 0 && untouched_active == 0, "inactive game was modified");
    base = {};
    std::fill(std::begin(base.candidates), std::end(base.candidates), 1);
    base.target = Solution(v, "ABACK");
    base.guesses = 0;
    base.candidates[Solution(v, "ADOBE")] = 1;
    r = run(base, 1, Action(v, "ARISE"), -1);
    Require(r.first.candidates[base.target] && !r.first.candidates[Solution(v, "ADOBE")], "candidate filtering failed");
    cudaFree(logits);
    cudaFree(active);
    cudaFree(g);
    cudaFree(feedback);
    cudaFree(s);
    cudaFree(a);
}

void StartTest(const f::Vocabulary &v) {
    gs::Slab slab;
    CheckCuda(slab.Create(2), "slab");
    gs::Slot *allocated = nullptr;
    CheckCuda(cudaMalloc(&allocated, sizeof(gs::Slot)), "allocated slot");
    AllocateOne<<<1, 1, 0, gStream>>>(slab.view(), allocated);
    CheckCuda(cudaStreamSynchronize(gStream), "allocate slot");
    gs::Slot slot = gs::kInvalidSlot;
    Copy(allocated, &slot, sizeof(slot), cudaMemcpyDeviceToHost);
    CheckCuda(cudaStreamSynchronize(gStream), "slot copy");
    Require(slot != gs::kInvalidSlot, "slab allocation failed");
    std::array<gs::Slot, 2> slots{slot, slot};
    f::Word *actions = nullptr, *solutions = nullptr;
    CheckCuda(cudaMalloc(&actions, sizeof(v.actions)), "start actions");
    CheckCuda(cudaMalloc(&solutions, sizeof(v.solutions)), "start solutions");
    Copy(v.actions.data(), actions, sizeof(v.actions), cudaMemcpyHostToDevice);
    Copy(v.solutions.data(), solutions, sizeof(v.solutions), cudaMemcpyHostToDevice);
    gs::Slot *ds = nullptr;
    f::Game *dg = nullptr;
    int *da = nullptr;
    const float **dp = nullptr;
    CheckCuda(cudaMalloc(&ds, 8), "slots");
    CheckCuda(cudaMalloc(&dg, 2 * sizeof(f::Game)), "games");
    CheckCuda(cudaMalloc(&da, 8), "active");
    CheckCuda(cudaMalloc(&dp, 2 * sizeof(float *)), "params");
    Copy(slots.data(), ds, 8, cudaMemcpyHostToDevice);
    f::Tables t{actions, solutions, nullptr, v.training_solutions.data(), nullptr};
    std::uint16_t *train = nullptr;
    CheckCuda(cudaMalloc(&train, sizeof(v.training_solutions)), "training");
    Copy(v.training_solutions.data(), train, sizeof(v.training_solutions), cudaMemcpyHostToDevice);
    t.training_solutions = train;
    CheckCuda(f::StartGames(t, slab.view(), ds, 2108, 2, dg, da, dp, gStream), "StartGames");
    CheckCuda(cudaStreamSynchronize(gStream), "Start sync");
    std::array<f::Game, 2> games{};
    Copy(dg, games.data(), sizeof(games), cudaMemcpyDeviceToHost);
    CheckCuda(cudaStreamSynchronize(gStream), "start result copy");
    Require(games[0].organism == 0 && games[1].organism == 1 && games[0].target == v.training_solutions[2108] &&
                games[1].target == v.training_solutions[0],
            "start boundary failed");
    Require(games[0].candidates[0] && games[1].candidates[m::kNumSolutions - 1], "candidate initialization failed");
    cudaFree(train);
    cudaFree(solutions);
    cudaFree(actions);
    cudaFree(allocated);
    cudaFree(dp);
    cudaFree(da);
    cudaFree(dg);
    cudaFree(ds);
    CheckCuda(slab.Destroy(), "slab destroy");
}
} // namespace

int main() {
    SelectTestGpu();
    CheckCuda(cudaStreamCreateWithFlags(&gStream, cudaStreamNonBlocking), "stream create");
    auto v = f::LoadVocabulary(WORD_DATA_DIR);
    EncodeGolden(v);
    AdvanceTests(v);
    StartTest(v);
    CheckCuda(cudaStreamDestroy(gStream), "stream destroy");
    std::puts("CUDA fitness game primitives passed");
}
