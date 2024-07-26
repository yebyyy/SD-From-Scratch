# SD-From-Scratch

Follows Umar Jamil's [lecture](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=397s) and the papers [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) and [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752) to build Stable Diffusion Model from scratch.

The Stable diffusion model includes **Variational Autoencoder(VAE)** from the paper [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114), **Contrastive Language–Image Pre-training(CLIP)** proposed by OpenAI in the paper [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020), **U-Net** from the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597), Self Attention and Cross Attention from the [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper, 

Weights and the tokenizer are loaded from the HuggingFace [RunwayML Stavle Diffusion V1,5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).
