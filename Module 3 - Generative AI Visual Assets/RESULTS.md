# Module 3 training and generation report

## Week 7: VAE

The VAE used the Module 2 generated screenshot dataset at 64×64 resolution. Scenario separated splits contained 144 training, 48 validation, and 48 test images. An Adam optimizer trained the encoder and decoder for 12 epochs with seed 42. The checkpoint with the lowest validation loss was used for visual review and the test calculation.

| Run | Beta | Best validation loss | Test loss | Visual observation |
|---|---:|---:|---:|---|
| Default | 0.05 | 72.01 | 69.88 | Recognizable broad layout in reconstructions; text is blurred |
| Stronger KL | 1.0 | 97.07 | 96.39 | More regular-looking samples; reconstructions lose additional detail |

The objective is summed pixel MSE per image plus beta times KL divergence. The two runs optimize different objectives, so their total loss numbers should not be treated as a direct image quality ranking. The training and validation curves both decreased across the 12 epochs; these curves do not show a clear train/validation divergence in this short run. The generated sample grids remain too blurry for a usable support asset. In particular, small text and icons are not legible. A tiny dataset, 64×64 resizing, and pixelwise loss limit fidelity.

Reconstruction starts from an actual image and passes its encoded mean to the decoder. Prior sampling starts from a new random latent vector; the resulting image is generated without a source image. This distinction is visible in the two saved grids.

Files: `artifacts/vae/` and `artifacts/vae_beta1/` each contain a checkpoint, `history.csv`, `metrics.json`, `training_curves.png`, `reconstructions.png`, and `samples.png`.

## Week 8: diffusion review

The configured diffusion workflow uses `segmind/tiny-sd`, two generic support-asset prompts, and two step/guidance settings per prompt. The same seed is reused across settings for a given prompt. A run saves the exact configuration, output PNGs, a contact sheet, and a `review.json` record for acceptance decisions and notes. Quality is assessed by looking at the images; no automatic score here proves prompt fidelity or suitability.

The completed 128×128 quick check generated two images with the same support-inbox prompt and seed 42. At 4 steps/guidance 4, the image was blurry; at 8 steps/guidance 7, it still did not clearly show a support inbox. Both were marked `rejected` with reasons in `artifacts/diffusion_runs/20261030T003442437924Z-532fc5bd/review.json`. This is one prompt and seed, so it does not establish a general ranking of the settings. The initial 512×512 CPU check was interrupted during inference; its config-only folder is an incomplete run.

This pretrained model's repository provides PyTorch `.bin` weights, and Diffusers reported pickle loading during the local check. Only load model weights from a source you trust. Review the model card and license before distributing outputs. Generated visuals can contain unwanted lettering, logo-like marks, or biased content, and they are not authentic support screenshots.

The training and test images were inspected while developing this module, so these measurements are instructional observations rather than an untouched final benchmark. No customer data was used.
