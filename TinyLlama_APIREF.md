# TinyLlama API Reference

Complete API documentation for integrating TinyLlama into your applications.

---

> **📝 Portfolio Documentation Sample**
> 
> This documentation was created by Sarah Chan as a demonstration of technical writing skills for potential employers. It is not part of the official TinyLlama project or community. For official TinyLlama documentation, please visit the [TinyLlama GitHub repository](https://github.com/jzhang38/TinyLlama).

---

## Overview

The TinyLlama API provides programmatic access to the TinyLlama language model through the Transformers library. This reference covers all public methods, parameters, and configuration options for model loading, text generation, and inference optimization.

**Supported Interfaces:**
- Python (Transformers library)
- REST API (via inference servers)
- Command-line interface

**Version Coverage:** TinyLlama 1.1B (all checkpoints and variants)

---

## Table of Contents

- [Model Loading](#model-loading)
- [Text Generation](#text-generation)
- [Configuration](#configuration)
- [Tokenization](#tokenization)
- [Optimization](#optimization)
- [Error Handling](#error-handling)
- [REST API](#rest-api)

---

## Model Loading

### AutoModelForCausalLM.from_pretrained()

Load a TinyLlama model for causal language modeling tasks.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | Required | Model identifier or path. Use `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` for the chat model or `"TinyLlama/TinyLlama-1.1B-intermediate-step-480k-1T"` for base checkpoints. |
| `torch_dtype` | `torch.dtype` | `torch.float32` | Data type for model weights. Use `torch.float16` or `torch.bfloat16` for reduced memory usage. |
| `device_map` | `str` or `dict` | `None` | Device placement strategy. Options: `"auto"`, `"cpu"`, `"cuda"`, or custom device mapping. |
| `load_in_8bit` | `bool` | `False` | Enable 8-bit quantization to reduce memory footprint by ~75%. Requires `bitsandbytes` library. |
| `load_in_4bit` | `bool` | `False` | Enable 4-bit quantization for maximum memory efficiency. Requires `bitsandbytes` library. |
| `trust_remote_code` | `bool` | `False` | Allow execution of custom modeling code from the repository. |
| `revision` | `str` | `"main"` | Specific model version or git branch/tag to load. |
| `cache_dir` | `str` | `None` | Directory to cache downloaded models. Defaults to `~/.cache/huggingface/`. |
| `low_cpu_mem_usage` | `bool` | `False` | Reduce CPU RAM usage during model loading. |

#### Returns

`AutoModelForCausalLM` — Loaded language model ready for inference.

#### Examples

**Basic Loading:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
```

**GPU Acceleration with Half Precision:**
```python
import torch

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

**8-bit Quantization:**
```python
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_8bit=True,
    device_map="auto"
)
```

#### Exceptions

- `OSError`: Model files not found or network error during download
- `ValueError`: Invalid model identifier or incompatible parameters
- `RuntimeError`: Insufficient GPU memory or CUDA errors

---

### AutoTokenizer.from_pretrained()

Load the tokenizer associated with TinyLlama models.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pretrained_model_name_or_path` | `str` | Required | Model identifier. Must match the model used for inference. |
| `use_fast` | `bool` | `True` | Use the fast Rust-based tokenizer implementation when available. |
| `padding_side` | `str` | `"right"` | Side for padding tokens. Options: `"left"` or `"right"`. |
| `truncation_side` | `str` | `"right"` | Side for truncation. Options: `"left"` or `"right"`. |
| `cache_dir` | `str` | `None` | Custom cache directory for tokenizer files. |

#### Returns

`AutoTokenizer` — Tokenizer instance compatible with TinyLlama.

#### Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    use_fast=True
)
```

---

## Text Generation

### model.generate()

Generate text sequences using the loaded TinyLlama model.

```python
outputs = model.generate(
    inputs,
    **generation_config
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `torch.Tensor` | Required | Input token IDs. Shape: `(batch_size, sequence_length)`. |
| `max_length` | `int` | `None` | Maximum total length of generated sequence (input + output). |
| `max_new_tokens` | `int` | `None` | Maximum number of new tokens to generate. Takes precedence over `max_length`. |
| `min_length` | `int` | `0` | Minimum total length of generated sequence. |
| `min_new_tokens` | `int` | `None` | Minimum number of new tokens to generate. |
| `do_sample` | `bool` | `False` | Enable sampling for non-deterministic generation. If `False`, uses greedy decoding. |
| `temperature` | `float` | `1.0` | Sampling temperature. Higher values (e.g., 1.5) increase randomness. Lower values (e.g., 0.3) make output more focused. Only applies when `do_sample=True`. |
| `top_k` | `int` | `50` | Limit sampling to top-k tokens by probability. Set to 0 to disable. |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold. Only tokens with cumulative probability ≤ `top_p` are considered. Range: 0.0-1.0. |
| `repetition_penalty` | `float` | `1.0` | Penalty for repeating tokens. Values > 1.0 discourage repetition. Typical range: 1.0-1.5. |
| `length_penalty` | `float` | `1.0` | Exponential penalty for length. Values > 1.0 encourage longer sequences, < 1.0 encourage shorter. |
| `num_beams` | `int` | `1` | Number of beams for beam search. Higher values improve quality but increase computation. |
| `num_return_sequences` | `int` | `1` | Number of independent sequences to generate. Must be ≤ `num_beams`. |
| `early_stopping` | `bool` | `False` | Stop beam search when `num_beams` complete sequences are found. |
| `pad_token_id` | `int` | `None` | Token ID for padding. Defaults to tokenizer's pad token. |
| `eos_token_id` | `int` or `list` | `None` | Token ID(s) that signal end of sequence. Generation stops when produced. |
| `no_repeat_ngram_size` | `int` | `0` | Prevent repeating n-grams of this size. 0 = disabled. |
| `bad_words_ids` | `list` | `None` | List of token IDs that should not be generated. |
| `output_scores` | `bool` | `False` | Return generation scores (logits) for each step. |
| `return_dict_in_generate` | `bool` | `False` | Return a `GenerateOutput` dictionary instead of just token IDs. |

#### Returns

**Default:** `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`  
**With `return_dict_in_generate=True`:** `GenerateOutput` object containing:
- `sequences`: Generated token IDs
- `scores`: Generation scores (if `output_scores=True`)
- `hidden_states`: Hidden states (if `output_hidden_states=True`)
- `attentions`: Attention weights (if `output_attentions=True`)

#### Examples

**Basic Generation (Greedy):**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

**Sampling with Temperature:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=50
)
```

**Beam Search:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,
    early_stopping=True,
    num_return_sequences=3
)
```

**Repetition Control:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)
```

#### Common Parameter Combinations

**Creative Writing:**
```python
temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.2
```

**Factual/Precise Output:**
```python
temperature=0.3, top_p=0.9, top_k=40, repetition_penalty=1.1
```

**Balanced General Use:**
```python
temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.15
```

---

## Tokenization

### tokenizer.encode()

Convert text to token IDs.

```python
token_ids = tokenizer.encode(
    text,
    add_special_tokens=True,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` or `list[str]` | Required | Input text to tokenize. |
| `add_special_tokens` | `bool` | `True` | Add model-specific special tokens (BOS, EOS). |
| `padding` | `bool` or `str` | `False` | Padding strategy. Options: `True`, `"longest"`, `"max_length"`, `False`. |
| `truncation` | `bool` | `False` | Enable truncation to `max_length`. |
| `max_length` | `int` | `None` | Maximum sequence length. |
| `return_tensors` | `str` | `None` | Return tensor type. Options: `"pt"` (PyTorch), `"np"` (NumPy), `None` (list). |

#### Returns

`list[int]` — Token IDs, or tensors if `return_tensors` is specified.

#### Example

```python
# Single sequence
text = "Hello, how are you?"
token_ids = tokenizer.encode(text, add_special_tokens=True)

# With tensors
inputs = tokenizer(text, return_tensors="pt")
print(inputs.input_ids)  # PyTorch tensor
```

---

### tokenizer.decode()

Convert token IDs back to text.

```python
text = tokenizer.decode(
    token_ids,
    skip_special_tokens=True,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_ids` | `list[int]` or `torch.Tensor` | Required | Token IDs to decode. |
| `skip_special_tokens` | `bool` | `False` | Remove special tokens (BOS, EOS, PAD) from output. |
| `clean_up_tokenization_spaces` | `bool` | `True` | Clean up extra spaces in output. |

#### Returns

`str` — Decoded text string.

#### Example

```python
token_ids = [1, 15043, 29892, 920, 526, 366, 29973, 2]
text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(text)  # "Hello, how are you?"
```

---

### tokenizer.batch_encode_plus()

Encode multiple sequences efficiently.

```python
batch_encoding = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs,
    **kwargs
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_text_or_text_pairs` | `list[str]` | Required | List of texts to encode. |
| `add_special_tokens` | `bool` | `True` | Add special tokens to each sequence. |
| `padding` | `str` | `False` | Padding strategy: `"longest"`, `"max_length"`, or `False`. |
| `truncation` | `bool` | `False` | Truncate sequences to `max_length`. |
| `max_length` | `int` | `None` | Maximum sequence length. |
| `return_tensors` | `str` | `None` | Return format: `"pt"`, `"np"`, or `None`. |

#### Returns

`BatchEncoding` — Dictionary containing `input_ids`, `attention_mask`, and optional `token_type_ids`.

#### Example

```python
texts = [
    "First example text.",
    "Second example text.",
    "Third example text."
]

batch = tokenizer.batch_encode_plus(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

print(batch.input_ids.shape)  # torch.Size([3, 128])
```

---

## Configuration

### GenerationConfig

Pre-configure generation parameters for reusable settings.

```python
from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

outputs = model.generate(**inputs, generation_config=generation_config)
```

#### Common Configurations

**Creative Writing Config:**
```python
creative_config = GenerationConfig(
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.2,
    do_sample=True
)
```

**Precise Factual Config:**
```python
factual_config = GenerationConfig(
    max_new_tokens=150,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True
)
```

**Chat Config:**
```python
chat_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)
```

---

## Optimization

### Memory Optimization

#### 8-bit Quantization

Reduce memory usage by ~75% with minimal quality loss.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_8bit=True,
    device_map="auto"
)
```

**Requirements:**
```bash
pip install bitsandbytes accelerate
```

**Memory Impact:**
- Full precision (FP32): ~4.4 GB
- Half precision (FP16): ~2.2 GB
- 8-bit quantization: ~1.1 GB
- 4-bit quantization: ~637 MB

---

#### 4-bit Quantization

Maximum memory efficiency for deployment on resource-constrained devices.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

### Inference Speed Optimization

#### GPU Acceleration

```python
import torch

# Move model to GPU
model = model.to("cuda")

# Use half precision
model = model.half()

# Enable PyTorch optimizations
torch.backends.cudnn.benchmark = True
```

---

#### Batch Processing

Process multiple inputs simultaneously for better throughput.

```python
prompts = [
    "Explain machine learning:",
    "What is quantum computing?",
    "Describe neural networks:"
]

# Tokenize batch
inputs = tokenizer(prompts, return_tensors="pt", padding=True)

# Generate for all prompts at once
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode results
results = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
```

---

#### Speculative Decoding

Use TinyLlama as a draft model to accelerate larger models.

```python
from transformers import AutoModelForCausalLM

# Load both models
large_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
draft_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Generate with speculative decoding
outputs = large_model.generate(
    **inputs,
    assistant_model=draft_model,
    max_new_tokens=100,
    do_sample=True
)
```

**Speed Improvement:** 2-3x faster than standard generation for Llama-2-7B.

---

## Error Handling

### Common Exceptions

#### OutOfMemoryError

**Cause:** Model too large for available GPU memory.

**Solutions:**
```python
# Solution 1: Use quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Solution 2: Use CPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)

# Solution 3: Reduce batch size
outputs = model.generate(**inputs, batch_size=1)
```

---

#### ValueError: Invalid generation parameters

**Cause:** Incompatible parameter combinations.

**Example:**
```python
# Error: num_return_sequences > num_beams
outputs = model.generate(**inputs, num_beams=3, num_return_sequences=5)
```

**Fix:**
```python
outputs = model.generate(**inputs, num_beams=5, num_return_sequences=3)
```

---

#### OSError: Model not found

**Cause:** Invalid model identifier or network issues.

**Solutions:**
```python
# Solution 1: Check model name
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Correct spelling
)

# Solution 2: Use local path
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/local/model"
)

# Solution 3: Specify cache directory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="/custom/cache/path"
)
```

---

## REST API

For production deployments, serve TinyLlama through a REST API using inference servers.

### POST /v1/completions

Generate text completions.

**Endpoint:**
```
POST https://api.example.com/v1/completions
```

**Headers:**
```
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
```

**Request Body:**

```json
{
  "model": "tinyllama-1.1b-chat",
  "prompt": "Explain quantum computing:",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": ["\n\n", "Human:", "Assistant:"]
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | Yes | Model identifier. |
| `prompt` | `string` | Yes | Input text prompt. |
| `max_tokens` | `integer` | No | Maximum tokens to generate (default: 50). |
| `temperature` | `float` | No | Sampling temperature 0.0-2.0 (default: 1.0). |
| `top_p` | `float` | No | Nucleus sampling threshold (default: 1.0). |
| `n` | `integer` | No | Number of completions to generate (default: 1). |
| `stop` | `string` or `array` | No | Sequences where generation stops. |
| `presence_penalty` | `float` | No | Penalty for token presence -2.0 to 2.0 (default: 0). |
| `frequency_penalty` | `float` | No | Penalty for token frequency -2.0 to 2.0 (default: 0). |

**Response:**

```json
{
  "id": "cmpl-7a9b8c7d",
  "object": "text_completion",
  "created": 1677858242,
  "model": "tinyllama-1.1b-chat",
  "choices": [
    {
      "text": "Quantum computing harnesses quantum mechanics principles...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 100,
    "total_tokens": 104
  }
}
```

**Status Codes:**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request parameters |
| 401 | Authentication failed |
| 429 | Rate limit exceeded |
| 500 | Server error |

---

### POST /v1/chat/completions

Generate chat-style responses with conversation context.

**Request Body:**

```json
{
  "model": "tinyllama-1.1b-chat",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain neural networks."}
  ],
  "temperature": 0.7,
  "max_tokens": 150
}
```

**Response:**

```json
{
  "id": "chatcmpl-8d9e0f1g",
  "object": "chat.completion",
  "created": 1677858365,
  "model": "tinyllama-1.1b-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Neural networks are computing systems inspired by biological neural networks..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 87,
    "total_tokens": 111
  }
}
```

---

## Code Examples

### Complete Chat Application

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class TinyLlamaChat:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize chat model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.generation_config = GenerationConfig(
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True
        )
        
        self.conversation_history = []
    
    def chat(self, user_message):
        """Generate a response to user message."""
        # Add user message to history
        self.conversation_history.append(f"User: {user_message}")
        
        # Create prompt from history
        prompt = "\n".join(self.conversation_history) + "\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_response.split("Assistant:")[-1].strip()
        
        # Add to history
        self.conversation_history.append(f"Assistant: {assistant_response}")
        
        return assistant_response
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []

# Usage
chatbot = TinyLlamaChat()
response = chatbot.chat("What is machine learning?")
print(response)
```

---

### Streaming Generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt")

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Generate in separate thread
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Stream output
print(prompt, end="")
for new_text in streamer:
    print(new_text, end="", flush=True)
print()
```

---

## Performance Metrics

### Latency Benchmarks

**First Token Latency** (time to first generated token):

| Hardware | Precision | Latency |
|----------|-----------|---------|
| NVIDIA RTX 4090 | FP16 | 15 ms |
| NVIDIA RTX 3090 | FP16 | 22 ms |
| Apple M2 Max | FP32 | 45 ms |
| Intel i9-13900K | FP32 | 180 ms |

**Throughput** (tokens per second):

| Hardware | Precision | Tokens/sec |
|----------|-----------|------------|
| NVIDIA RTX 4090 | FP16 | 145 |
| NVIDIA RTX 3090 | FP16 | 105 |
| Apple M2 Max | FP32 | 48 |
| Intel i9-13900K | FP32 | 28 |

---

## Version History

| Version | Release Date | Changes |
|---------|-------------|---------|
| 1.1B-Chat-v1.0 | Dec 2023 | Initial chat model release |
| 1.1B-intermediate-step-480k-1T | Oct 2023 | 1 trillion token checkpoint |
| 1.1B-intermediate-step-715k-1.5T | Nov 2023 | 1.5 trillion token checkpoint |
| 1.1B-intermediate-step-955k-2T | Dec 2023 | 2 trillion token checkpoint |

---

## Support and Resources

- **API Issues**: [Report API problems](https://github.com/project/repo/issues)
- **Model Registry**: [Browse available models](https://modelregistry.example.com)
- **Developer Forum**: [Ask questions](https://community.example.com/developers)
- **Framework Documentation**: [Transformers library docs](https://docs.framework.example.com)

---

**Last Updated:** January 2026  
**API Version:** 1.0
