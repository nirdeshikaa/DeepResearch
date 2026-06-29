# DeepResearch LoRA Fine-Tuning Script
# Run this on Google Colab with GPU runtime (T4 or A100)
# Runtime > Change runtime type > T4 GPU

# ============================================================
# STEP 1: Install dependencies
# ============================================================
# !pip install -q transformers peft datasets trl bitsandbytes accelerate

# ============================================================
# STEP 2: Imports
# ============================================================
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

TRAINING_DATA = [
    {"instruction": "What is Retrieval-Augmented Generation (RAG)?", "response": "Retrieval-Augmented Generation (RAG) is a framework that combines retrieval-based and generation-based approaches. It first retrieves relevant documents from a knowledge base using semantic search, then conditions a language model on the retrieved context to generate accurate, grounded answers. RAG reduces hallucination by grounding responses in retrieved evidence rather than relying solely on parametric memory."},
    {"instruction": "How does LoRA reduce memory requirements for fine-tuning?", "response": "LoRA (Low-Rank Adaptation) reduces memory by freezing the original model weights and injecting small trainable rank-decomposition matrices into each transformer layer. Instead of updating all parameters, LoRA only trains matrices A and B where the weight update is W = W0 + BA. This reduces trainable parameters by up to 10,000x and GPU memory by 3x while achieving comparable performance to full fine-tuning."},
    {"instruction": "What causes hallucination in large language models?", "response": "Hallucination in LLMs occurs when models generate fluent but factually incorrect or fabricated content. Key causes include: (1) training data gaps where the model lacks knowledge of specialised domains, (2) overconfident generation that prioritises language fluency over factual accuracy, (3) retrieval failures in RAG systems where irrelevant chunks mislead the generator, and (4) distributional shift between training and inference domains."},
    {"instruction": "What is FAISS and how does it work?", "response": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vector embeddings. It works by building an index of vectors and supporting approximate nearest-neighbour search using techniques like inverted file indexes and product quantization. FAISS supports millions of vectors on consumer hardware with sub-second query times, making it ideal for RAG retrieval pipelines."},
    {"instruction": "What is the difference between RAG and fine-tuning?", "response": "RAG and fine-tuning address different problems. RAG dynamically retrieves external knowledge at inference time, making it ideal for tasks requiring up-to-date or domain-specific information without retraining. Fine-tuning updates model weights on task-specific data, improving the model's style, format, and domain vocabulary. For academic paper synthesis, combining both yields the best results."},
    {"instruction": "What is QLoRA?", "response": "QLoRA is an extension of LoRA that combines quantisation with low-rank adaptation. It loads the base model in 4-bit precision using NF4 quantisation while keeping the LoRA adapters in higher precision. This allows fine-tuning of 7-13 billion parameter models on a single consumer GPU with 24GB VRAM."},
    {"instruction": "How does dense passage retrieval work?", "response": "Dense passage retrieval uses neural embeddings to represent both queries and documents in a shared vector space. A query encoder converts the user question into a dense vector, and a document encoder converts passages into dense vectors stored in a FAISS index. At retrieval time, cosine similarity identifies the top-k most relevant passages."},
    {"instruction": "What is instruction tuning?", "response": "Instruction tuning is a fine-tuning technique where a pre-trained language model is trained on (instruction, response) pairs to follow natural language instructions. It improves zero-shot generalisation by teaching the model to understand and execute diverse task descriptions."},
    {"instruction": "What are the limitations of standard RAG pipelines?", "response": "Standard RAG pipelines have several limitations: (1) retrieval failures where irrelevant chunks mislead the generator, (2) vocabulary mismatch between query and document embeddings in specialised domains, (3) context window limitations, (4) lack of reasoning across multiple retrieved documents, and (5) generation-phase hallucination."},
    {"instruction": "How is Precision@5 used to evaluate retrieval quality?", "response": "Precision@5 measures retrieval quality by checking how many of the top-5 retrieved documents are relevant to the query. A score of 1.0 means all 5 retrieved chunks are relevant. For DeepResearch, a target Precision@5 of 0.70 was set, and the system achieved 0.96."},
]

def format_sample(sample):
    return f"""### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"""

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(model, tokenizer):
    formatted = [format_sample(s) for s in TRAINING_DATA]
    dataset = Dataset.from_dict({"text": formatted})
    training_args = TrainingArguments(
        output_dir="./deepresearch-lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_ratio=0.03,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained("./deepresearch-lora-final")
    tokenizer.save_pretrained("./deepresearch-lora-final")
    print("✅ LoRA adapter saved!")

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model()
    print("Applying LoRA...")
    model = apply_lora(model)
    print("Training...")
    train(model, tokenizer)
EOFcat > ~/deepresearch/lora_finetune.py << 'EOF'
# DeepResearch LoRA Fine-Tuning Script
# Run this on Google Colab with GPU runtime (T4 or A100)
# Runtime > Change runtime type > T4 GPU

# ============================================================
# STEP 1: Install dependencies
# ============================================================
# !pip install -q transformers peft datasets trl bitsandbytes accelerate

# ============================================================
# STEP 2: Imports
# ============================================================
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

TRAINING_DATA = [
    {"instruction": "What is Retrieval-Augmented Generation (RAG)?", "response": "Retrieval-Augmented Generation (RAG) is a framework that combines retrieval-based and generation-based approaches. It first retrieves relevant documents from a knowledge base using semantic search, then conditions a language model on the retrieved context to generate accurate, grounded answers. RAG reduces hallucination by grounding responses in retrieved evidence rather than relying solely on parametric memory."},
    {"instruction": "How does LoRA reduce memory requirements for fine-tuning?", "response": "LoRA (Low-Rank Adaptation) reduces memory by freezing the original model weights and injecting small trainable rank-decomposition matrices into each transformer layer. Instead of updating all parameters, LoRA only trains matrices A and B where the weight update is W = W0 + BA. This reduces trainable parameters by up to 10,000x and GPU memory by 3x while achieving comparable performance to full fine-tuning."},
    {"instruction": "What causes hallucination in large language models?", "response": "Hallucination in LLMs occurs when models generate fluent but factually incorrect or fabricated content. Key causes include: (1) training data gaps where the model lacks knowledge of specialised domains, (2) overconfident generation that prioritises language fluency over factual accuracy, (3) retrieval failures in RAG systems where irrelevant chunks mislead the generator, and (4) distributional shift between training and inference domains."},
    {"instruction": "What is FAISS and how does it work?", "response": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search over dense vector embeddings. It works by building an index of vectors and supporting approximate nearest-neighbour search using techniques like inverted file indexes and product quantization. FAISS supports millions of vectors on consumer hardware with sub-second query times, making it ideal for RAG retrieval pipelines."},
    {"instruction": "What is the difference between RAG and fine-tuning?", "response": "RAG and fine-tuning address different problems. RAG dynamically retrieves external knowledge at inference time, making it ideal for tasks requiring up-to-date or domain-specific information without retraining. Fine-tuning updates model weights on task-specific data, improving the model's style, format, and domain vocabulary. For academic paper synthesis, combining both yields the best results."},
    {"instruction": "What is QLoRA?", "response": "QLoRA is an extension of LoRA that combines quantisation with low-rank adaptation. It loads the base model in 4-bit precision using NF4 quantisation while keeping the LoRA adapters in higher precision. This allows fine-tuning of 7-13 billion parameter models on a single consumer GPU with 24GB VRAM."},
    {"instruction": "How does dense passage retrieval work?", "response": "Dense passage retrieval uses neural embeddings to represent both queries and documents in a shared vector space. A query encoder converts the user question into a dense vector, and a document encoder converts passages into dense vectors stored in a FAISS index. At retrieval time, cosine similarity identifies the top-k most relevant passages."},
    {"instruction": "What is instruction tuning?", "response": "Instruction tuning is a fine-tuning technique where a pre-trained language model is trained on (instruction, response) pairs to follow natural language instructions. It improves zero-shot generalisation by teaching the model to understand and execute diverse task descriptions."},
    {"instruction": "What are the limitations of standard RAG pipelines?", "response": "Standard RAG pipelines have several limitations: (1) retrieval failures where irrelevant chunks mislead the generator, (2) vocabulary mismatch between query and document embeddings in specialised domains, (3) context window limitations, (4) lack of reasoning across multiple retrieved documents, and (5) generation-phase hallucination."},
    {"instruction": "How is Precision@5 used to evaluate retrieval quality?", "response": "Precision@5 measures retrieval quality by checking how many of the top-5 retrieved documents are relevant to the query. A score of 1.0 means all 5 retrieved chunks are relevant. For DeepResearch, a target Precision@5 of 0.70 was set, and the system achieved 0.96."},
]

def format_sample(sample):
    return f"""### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"""

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(model, tokenizer):
    formatted = [format_sample(s) for s in TRAINING_DATA]
    dataset = Dataset.from_dict({"text": formatted})
    training_args = TrainingArguments(
        output_dir="./deepresearch-lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        warmup_ratio=0.03,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained("./deepresearch-lora-final")
    tokenizer.save_pretrained("./deepresearch-lora-final")
    print("✅ LoRA adapter saved!")

if __name__ == "__main__":
    print("Loading model...")
    model, tokenizer = load_model()
    print("Applying LoRA...")
    model = apply_lora(model)
    print("Training...")
    train(model, tokenizer)
