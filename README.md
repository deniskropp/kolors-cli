# Kolors CLI

A command-line interface for generating images using the Kolors model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/deniskropp/kolors-cli.git
    cd kolors-cli
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install the CLI:
    ```bash
    pip install .
    ```

## Usage

Generate an image from a text prompt:
    ```bash
    kolors-cli --prompt "A beautiful sunset over the mountains"
    ```

Generate an image from a text prompt with an image adapter:
    ```bash
    kolors-cli --prompt "A beautiful sunset over the mountains" --ip_adapter_image "path/to/image.jpg"
    ```

Additional options:
- `--ip_adapter_scale`: Image influence scale (default: 0.5).
- `--negative_prompt`: Negative prompt (default: '').
- `--seed`: Seed for random number generation (default: 0).
- `--randomize_seed`: Randomize the seed (default: False).
- `--width`: Width of the generated image (default: 1024).
- `--height`: Height of the generated image (default: 1024).
- `--guidance_scale`: Guidance scale (default: 5.0).
- `--num_inference_steps`: Number of inference steps (default: 25).
- `--output`: Output file path (default: output.png).
