# Handwritten Text Recognition

C++ handwritten text recognition pipeline for scanned note images. The project combines line segmentation, connected-component analysis, neural letter recognition, bigram handling, and dictionary-based spell correction to turn handwritten page images into plain text.

<img src="docs/text_recognition_preview.png" alt="Example handwritten input" width="680">

Related projects: [mrdakj/letters](https://github.com/mrdakj/letters) trains the Python/TensorFlow letter models, and [mrdakj/line_segmentation](https://github.com/mrdakj/line_segmentation) contains the C++/OpenCV line-segmentation work integrated here.

## What it does

- Segments handwritten pages into text lines using an image-processing pipeline based on strip connections.
- Extracts connected components and classifies whether a component represents one letter or a two-letter bigram.
- Runs exported Keras models from C++ with `frugally-deep`.
- Uses a JamSpell language model and a word dictionary to improve recognition output.
- Supports debug output for inspecting intermediate segmentation and recognition steps.

## Tech Stack

- C++17
- OpenCV
- CMake
- frugally-deep for running exported neural-network models in C++
- JamSpell for spell correction
- nlohmann/json, Eigen, FunctionalPlus through the model inference stack

## Pipeline

```text
input image
  -> line segmentation
  -> connected-component extraction
  -> one-letter / bigram decision
  -> neural letter prediction
  -> word assembly
  -> optional dictionary correction
  -> text output
```

## Repository Layout

```text
.
|-- images/          # handwritten input examples
|-- docs/            # README preview assets
|-- include/         # image utilities and line segmentation code
|-- model/           # exported recognition models and JamSpell language model
|-- out/             # generated recognition outputs
|-- dictionary/      # English word dictionary used by correction step
|-- jamspell/        # local JamSpell integration
|-- contrib/         # third-party support libraries
`-- main.cpp         # recognition pipeline and CLI entry point
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

Recognize one image:

```bash
./main ../images/1.png dictionary
```

Recognize one image and keep debug artifacts:

```bash
./main ../images/1.png dictionary debug
```

Recognize all images in a directory:

```bash
./main ../images
```

Outputs are written to `out/`. Files ending in `_dictionary.txt` include dictionary/spell-correction post-processing.

