# Kolors CLI

A command-line interface for generating images using the Kolors model. This tool leverages the power of the Kolors model to create images based on text prompts, with optional image prompts for more detailed and specific image generation.

## Features

- **Text-to-Image Generation**: Generate images from text prompts.
- **Image-to-Image Generation**: Use an existing image as a prompt to generate a new image.
- **Customization**: Adjust parameters like image size, guidance scale, and number of inference steps.
- **Random Seed**: Generate different images by randomizing the seed.

## Installation

To install the Kolors CLI, you can use pip:

    ```bash
    git clone deniskropp/kolors-cli.git
    cd kolors-cli
    pipx install -e .
    ```

## Usage

### Basic Usage

Generate an image from a text prompt:

    ```bash
    kolors-cli --prompt "A beautiful landscape with mountains and a river"
    ```

### Advanced Usage

Generate an image with an image prompt:

    ```bash
    kolors-cli --prompt "A beautiful landscape with mountains and a river" --ip_adapter_image "path/to/image.jpg"
    ```

### Full Options

    ```bash
    kolors-cli --prompt "A beautiful landscape with mountains and a river" \
           --ip_adapter_image "path/to/image.jpg" \
           --ip_adapter_scale 0.5 \
           --negative_prompt "blurry, low quality" \
           --seed 42 \
           --randomize_seed \
           --width 1024 \
           --height 1024 \
           --guidance_scale 5.0 \
           --num_inference_steps 25 \
           --output "output.png"
    ```

### Options

- `--prompt`: The text prompt for image generation.
- `--ip_adapter_image`: Path to the image prompt (optional).
- `--ip_adapter_scale`: Image influence scale (default: 0.5).
- `--negative_prompt`: Negative prompt (default: '').
- `--seed`: Seed for random number generation (default: 0).
- `--randomize_seed`: Randomize the seed (default: False).
- `--width`: Width of the generated image (default: 1024).
- `--height`: Height of the generated image (default: 1024).
- `--guidance_scale`: Guidance scale (default: 5.0).
- `--num_inference_steps`: Number of inference steps (default: 25).
- `--output`: Output file path (default: output.png).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## Contact

For any questions or feedback, you can reach out to the author at dok@directfb1.org.

## Acknowledgments

- The Kolors model and its components.
- The Hugging Face team for their excellent libraries and resources.
- The community for their support and contributions.

Enjoy generating images with Kolors CLI!
