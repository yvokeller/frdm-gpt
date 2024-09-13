# Fine-tuning Configuration

This directory contains configuration files for fine-tuning language models using MLX.

## How to Run

To fine-tune the model using the LoRA (Low-Rank Adaptation) technique, use the following command:

```bash
mlx_lm.lora --config ft-configs/frdm-Llama-3.1-8B-Write-Beat-to-Prose-v1.yaml
```

## Configuration File

The `ft-configs/frdm-Llama-3.1-8B-Write-Beat-to-Prose-v1` file contains the specific configuration for fine-tuning the frdm-Llama-3.1-8B-Write-Beat-to-Prose-v1 model using LoRA. Make sure this file is present in the specified location before running the command.

## Additional Information

- Ensure you have the `mlx_lm` tool installed and properly set up in your environment.
- For more details on the configuration options and their meanings, refer to the MLX documentation.

## How to Generate

For inference, you can use the following command:

```bash
mlx_lm.generate --model models/frdm-Llama-3.1-8B-Write --adapter-path models/frdm-Llama-3.1-8B-Write-Beat-to-Prose-v1 --prompt "Still alive?"
```
