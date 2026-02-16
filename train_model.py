"""
Training script for fine-tuning the AI Voice Detector.
Usage: python train_model.py --dataset_path ./my_dataset --epochs 3

Dataset Structure:
/my_dataset
    /real
        human_audio1.mp3
        human_audio2.wav
    /fake
        ai_audio1.mp3
        ai_audio2.wav
"""
import os
import argparse
import torch
import numpy as np
import librosa
from datasets import load_dataset, Audio, ClassLabel, DatasetDict, Dataset
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from app.config import settings

# Configuration
MODEL_NAME = settings.MODEL_NAME
SAMPLE_RATE = settings.SAMPLE_RATE
OUTPUT_DIR = "./voice_detection_model_finetuned"

def setup_dataset(dataset_path):
    """
    Load dataset from folders
    """
    data = []
    
    # Check if directories exist
    real_dir = os.path.join(dataset_path, "real")
    fake_dir = os.path.join(dataset_path, "fake")
    
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise ValueError(f"Dataset path must contain 'real' and 'fake' subdirectories at {dataset_path}")

    # Load Real (Human) - Label 0
    for filename in os.listdir(real_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
            data.append({"audio": os.path.join(real_dir, filename), "label": 0})
            
    # Load Fake (AI) - Label 1
    for filename in os.listdir(fake_dir):
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
            data.append({"audio": os.path.join(fake_dir, filename), "label": 1})
            
    # Create Dataset
    ds = Dataset.from_list(data)
    
    # Split
    ds = ds.train_test_split(test_size=0.2)
    return ds

def preprocess_function(examples, feature_extractor):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate, 
        max_length=SAMPLE_RATE * 5, # 5 seconds max
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    return inputs

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

def main():
    parser = argparse.ArgumentParser(description="Train AI Voice Detector")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset containing 'real' and 'fake' folders")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    print(f"Loading base model: {MODEL_NAME}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    
    print(f"Preparing dataset from {args.dataset_path}...")
    dataset = setup_dataset(args.dataset_path)
    
    # Cast audio column
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    print("Preprocessing audio...")
    encoded_dataset = dataset.map(
        lambda x: preprocess_function(x, feature_extractor), 
        batched=True, 
        remove_columns=["audio"]
    )

    # Load Model
    # Label mapping: 0 -> HUMAN (Real), 1 -> AI_GENERATED (Fake)
    # Note: Check original model config to align permissions if needed, but for fine-tuning we can redefine.
    id2label = {0: "HUMAN", 1: "AI_GENERATED"}
    label2id = {"HUMAN": 0, "AI_GENERATED": 1}
    
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        label2id=label2id, 
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Training complete! Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    feature_extractor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
