def main():
    import torch
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    print("PyTorch version:", torch.__version__)
    print("Is CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()
