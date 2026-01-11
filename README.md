# Stable Diffusion From Scratch

A PyTorch implementation of Stable Diffusion built from scratch, following Umar Jamil's [tutorial](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=4284s).

## Project Overview

This project implements the core components of Stable Diffusion including:
- CLIP text encoder for processing prompts
- VAE encoder/decoder for working in latent space
- U-Net with cross-attention for conditioned prediction of image noise
- DDPM sampler for generating images from noise with fewer timesteps than training

## Setup Instructions

### Download Required Files

1. **Tokenizer files**: Download `vocab.json` and `merges.txt` from [stable-diffusion-v1-5/tokenizer](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) and save them in the `data` folder

2. **Model weights**: Download `v1-5-pruned-emaonly.ckpt` from [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and save it in the `data` folder


## Key Concepts Learned

- **Latent Diffusion Model**: Working in compressed latent space for efficiency
- **Classifier-Free Guidance**: Improving prompt adherence without a separate classifier
- **ELBO**: Evidence lower bound optimization for U-Net
- **Cross-Attention**: Conditioning image generation on text embeddings
- **U-Net Architecture**: Encoder-decoder with skip connections
