# Setup
1. clone the repository
2. install requirements `pip install -r requirements.txt`
3. (optional) install CUDA for NVIDIA GPU acceleration (https://developer.nvidia.com/cuda-downloads)

# Usage
## Basic training with default Parameters (default: 10 epochs)
```bash
python main.py
```

## Custom parameters for better control
```bash
python main.py --epochs 20 --seq-length 100 --batch-size 128 --data-file my_text.txt
```

## Example for large datasets/GPU training
```bash
python main.py --epochs 50 --batch-size 256 --seq-length 64
```

## Parameters:
- --epochs: Training iterations (default: 10)
- --seq-length: Input sequence length (default: 50, higher = longer context memory)
- --batch-size: Parallel sequences per batch (default: 64, increase for GPU)
- --data-file: Path to German training text (default: training_data_german.txt)



# Original Description
I made this becuase I was bored this is just very basic using torch and even supports cuda if avalable it will be auto picket

**IMPORTANT: This ai that comes with the premade training date is only in german you can do your own training data by deleting everyting in "text_training_data.txt" and putting your own thing into it so the ai will for example only speak in brainrot tearms**

If you want to delete your current ai then simply delete the .pt fileand rerun the program.

