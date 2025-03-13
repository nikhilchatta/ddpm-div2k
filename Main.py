from Diffusion.div2k_dataloader import get_div2k_dataloader
import torch
from Diffusion.Train import train

def main(model_config=None):
    # Determine the available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Default model configuration
    modelConfig = {
        "state": "train",  # or "eval"
        "epoch": 5,
        "batch_size": 32,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.0,
        "warmup_epoch": 5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 128,
        "grad_clip": 1.0,
        "device": device,
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8
    }

    # Override default config with provided model_config if available
    if model_config is not None:
        modelConfig.update(model_config)

    # Load DIV2K dataloader
    dataloader = get_div2k_dataloader(
        root='data/DIV2K_train_HR',
        image_size=modelConfig["img_size"],
        batch_size=modelConfig["batch_size"]
    )

    # Train or evaluate
    if modelConfig["state"] == "train":
        train(modelConfig, dataloader)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()
