# TinyLlama

A compact, efficient 1.1B parameter language model optimized for edge deployment and resource-constrained applications.

## Overview

TinyLlama is an open-source language model that brings the power of large language models to devices with limited computational resources. With only 1.1B parameters, it achieves impressive performance while maintaining a small memory footprint, making it ideal for deployment on edge devices, mobile applications, and scenarios where you need fast inference without cloud dependencies.

**Key Features:**
- **Compact Size**: Only 1.1B parameters (637 MB when 4-bit quantized)
- **Llama 2 Compatible**: Uses the same architecture and tokenizer as Llama 2
- **Production Ready**: Pretrained on 3 trillion tokens
- **Fast Inference**: Optimized for quick response times on consumer hardware
- **Offline Capable**: Run language model tasks without internet connectivity

## Use Cases

TinyLlama excels in scenarios where you need language model capabilities with constrained resources:

- **Edge Deployment**: Real-time translation, text generation, and chatbots on mobile and IoT devices
- **Speculative Decoding**: Speed up larger models by using TinyLlama to predict next tokens
- **Local Development**: Test and prototype language model applications without cloud costs
- **Privacy-Sensitive Applications**: Process sensitive data locally without external API calls
- **Offline Applications**: Enable AI features in environments without reliable internet access

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional, but recommended for faster inference)

### Installation

Install TinyLlama using pip:

```bash
pip install torch transformers
```

### Basic Usage

Generate text with TinyLlama in just a few lines of code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Generate text
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

**Expected output:**
```
Explain quantum computing in simple terms: Quantum computing uses the principles 
of quantum mechanics to process information. Unlike classical computers that use 
bits (0 or 1), quantum computers use qubits that can exist in multiple states 
simultaneously...
```

## Installation Options

### Option 1: Using Hugging Face Transformers (Recommended)

The easiest way to get started:

```bash
pip install transformers torch
```

### Option 2: Using llama.cpp for CPU Inference

For optimized CPU-only inference:

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build the project
make

# Download quantized TinyLlama model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Run inference
./main -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "Your prompt here"
```

### Option 3: Using Ollama

For a simple local setup with model management:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download and run TinyLlama
ollama run tinyllama
```

## Model Variants

TinyLlama is available in several configurations to suit different use cases:

| Model | Parameters | Use Case | Memory Required |
|-------|-----------|----------|-----------------|
| TinyLlama-1.1B | 1.1B | Full precision inference | ~4.5 GB |
| TinyLlama-1.1B-Chat | 1.1B | Conversational applications | ~4.5 GB |
| TinyLlama-1.1B-4bit | 1.1B | Quantized for lower memory | ~637 MB |
| TinyLlama-1.1B-intermediate | 1.1B | Various training checkpoints | ~4.5 GB |

## Configuration

### Adjusting Generation Parameters

Control the model's output behavior:

```python
outputs = model.generate(
    **inputs,
    max_length=200,           # Maximum tokens to generate
    temperature=0.7,          # Randomness (0.0 = deterministic, 1.0 = creative)
    top_p=0.9,               # Nucleus sampling threshold
    top_k=50,                # Consider only top K tokens
    do_sample=True,          # Enable sampling for varied outputs
    repetition_penalty=1.2   # Penalize repeated phrases
)
```

### Memory Optimization

For devices with limited RAM, enable 8-bit quantization:

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
```

## Performance

### Inference Speed

On consumer hardware, TinyLlama achieves:

- **NVIDIA RTX 3090**: ~100 tokens/second
- **Apple M2 CPU**: ~40 tokens/second  
- **Intel i7-12700K CPU**: ~25 tokens/second
- **Raspberry Pi 4**: ~2-3 tokens/second (with quantization)

### Model Benchmarks

TinyLlama performance on common reasoning tasks:

| Task | TinyLlama-1.1B | Llama-2-7B | Notes |
|------|----------------|------------|-------|
| HellaSwag | 59.2 | 77.2 | Common sense reasoning |
| OpenBookQA | 36.0 | 60.2 | Multi-choice Q&A |
| WinoGrande | 59.1 | 69.2 | Pronoun resolution |
| ARC-Easy | 55.3 | 76.3 | Science questions |

## Advanced Usage

### Chat Interface

Create a simple conversational interface:

```python
from transformers import pipeline

# Initialize chat pipeline
chat = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Format conversation
messages = [
    {"role": "user", "content": "What is the capital of France?"},
]

# Generate response
response = chat(messages, max_length=100)
print(response[0]['generated_text'])
```

### Speculative Decoding

Use TinyLlama to accelerate a larger model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load large and small models
large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

# Generate with speculative decoding
outputs = large_model.generate(
    **inputs,
    assistant_model=draft_model,
    do_sample=True,
    temperature=0.7
)
```

### Fine-tuning for Custom Tasks

Adapt TinyLlama to your specific use case:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## Troubleshooting

### Out of Memory Errors

**Problem**: Model fails to load with CUDA out of memory error.

**Solution**: Enable quantization or use CPU inference:
```python
# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_8bit=True,
    device_map="auto"
)
```

### Slow Generation on CPU

**Problem**: Text generation is too slow on CPU.

**Solution**: Use llama.cpp with quantized models for optimized CPU inference (see Installation Option 2).

### Poor Output Quality

**Problem**: Generated text is incoherent or repetitive.

**Solution**: Adjust generation parameters:
```python
outputs = model.generate(
    **inputs,
    temperature=0.8,         # Increase for more creativity
    top_p=0.95,             # Widen token selection
    repetition_penalty=1.3  # Reduce repetition
)
```

## Model Information

- **Architecture**: Llama 2
- **Parameters**: 1.1 billion
- **Training Data**: 3 trillion tokens (SlimPajama, StarCoder)
- **Vocabulary Size**: 32,000 tokens
- **Context Length**: 2,048 tokens
- **License**: Apache 2.0

## Resources

- **Repository**: [github.com/jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama)
- **Model Hub**: [huggingface.co/TinyLlama](https://huggingface.co/TinyLlama)
- **Paper**: [Training Details and Methodology](https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md)
- **Community**: [GitHub Discussions](https://github.com/jzhang38/TinyLlama/discussions)

## Contributing

We welcome contributions to TinyLlama documentation and examples. See our [Contributing Guide](CONTRIBUTING.md) for details.

## License

TinyLlama is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Acknowledgments

TinyLlama is built on the work of:
- Meta AI's Llama 2 architecture
- Lightning AI's lit-gpt framework
- The Flash Attention team

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/jzhang38/TinyLlama/issues)
- **Discussions**: Join the conversation on [GitHub Discussions](https://github.com/jzhang38/TinyLlama/discussions)
- **Documentation**: Full documentation at [docs.tinyllama.io](https://docs.tinyllama.io) (coming soon)

---

## About This Documentation

This documentation is part of an independent developer guide based on the archived TinyLlama project. It is intended to demonstrate how a small open-source language model can be documented for practical use by developers and systems integrators.
