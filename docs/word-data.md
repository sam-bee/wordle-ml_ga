# Word data

The five word lists in [`data/`](../data/) are byte-for-byte copies from the backprop project:

```text
../../talks/wordle/wordle-ml_machine-learning/data/
```

That path is relative to the repository root. The source checkout was at commit
`48ca50b1e19bb5feffd8a294e594afc6318a4b7d`, with no local changes to these files.

| File | Words | Role |
| --- | ---: | --- |
| `wordlist-action-space-4739.csv` | 4,739 | Guess vocabulary, indexed by policy output ID |
| `wordlist-valid-solutions-all-2309.csv` | 2,309 | Full answer vocabulary, indexed by candidate-mask ID |
| `wordlist-valid-solutions-train-2109.csv` | 2,109 | Training answers available for evolutionary fitness evaluation |
| `wordlist-valid-solutions-validation-100.csv` | 100 | Development evaluation answers |
| `wordlist-valid-solutions-test-100.csv` | 100 | Final evaluation holdout; keep out of fitness and tuning |

Each file contains one uppercase, five-letter ASCII word per line, in alphabetical order, with no header. The
train, validation, and test lists form a disjoint, complete partition of the answer vocabulary. Every answer is
also in the action vocabulary.

## Preserve vocabulary IDs

An action or solution ID is its zero-based line index in the corresponding full vocabulary. Preserve this ordering:
the policy's input and output weights depend on it. A word's position in a split file is not its global solution ID;
resolve it against the full answer vocabulary. Solution IDs and action IDs are separate index spaces.

The full candidate vocabulary includes all answers, including held-out answers. The split controls which target
answers are used to evaluate agents; it does not change the dimensions or ordering of the model's candidate mask.

The ordered action and solution SHA-256 hashes match the existing
[`policy manifest`](../tests/fixtures/policy/manifest.json). The source's canonical hash covers uppercase words with
one trailing LF per word, which is already the exact format of these copied files.

## Provenance and verification

The backprop project's `docs/data/overview-of-wordlists.md` traces the full solution and action vocabularies to
`wordle-ml_wordlists` commit `3f32b424c813d22bb2d73e8802e41b78e7d9ba68`. The action vocabulary contains all 2,309
answers plus 2,430 additional guesses selected using SUBTLEX-US word frequencies. The historical frequency cutoff
was not recorded. The three solution splits are preserved from the backprop project without reshuffling.

[`data/SHA256SUMS`](../data/SHA256SUMS) records each imported file's checksum. To check the copies in the development
container, run this from the repository root:

```sh
docker compose run --rm --no-deps -T cuda-dev sh -c 'cd data && sha256sum --check SHA256SUMS'
```

Import verification checked byte identity against the source files, word counts, format, alphabetical order,
uniqueness, split coverage/disjointness, answer membership in the action vocabulary, and model vocabulary hashes.
Checking holdout membership and identity does not score agents against it.

These lists supply the word data needed for gameplay and fitness evaluation. The backprop project's WDIT imitation
records contain teacher demonstrations for supervised learning and are not part of this import. Trained weights
already exist under `tests/fixtures/policy/` for inference regression checks; using them to seed evolution remains
undecided. Word loading, GPU encoding, and gameplay integration remain subsequent work.
