import pandas as pd
import torch
from transformers import pipeline
from typing import Optional
import re
import random


def inference(
    sequence_length: int,
    num_sequences: int = 100,
    output_path: str = "output.csv",
    model_path: str = "nferruz/ProtGPT2",
    seed: int = 42,
    top_k: int = 950,
    repetition_penalty: float = 1.2,
    temperature: float = 1.0,
    device: Optional[str] = None,
    batch_size: int = 10
):
    """
    ProtGPT2: Generation of Peptide Sequence

    Args:
        sequence_length
        num_sequences: number of sampling
        output_path
        model_path
        seed
        top_k: Top-k sampling
        repetition_penalty
        temperature
        device
        batch_size: inference batch

    Returns:
        pandas.DataFrame
    """
 
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading for model: {model_path}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    protgpt2 = pipeline(
        'text-generation',
        model=model_path,
        device_map="auto" if device == "cuda" else None,
        model_kwargs={"device_map": None} if device == "cpu" else {}
    )

    print(f"Start generation of sequence: {num_sequences} sequence, Sequence_length: {sequence_length}")

    # 1 token = about 4 amino acids
    max_new_tokens = max((sequence_length // 4) + 5, 10)

    all_sequences = []
    generated_count = 0
    max_iterations = num_sequences * 10
    iteration_count = 0
    candidate_sequences = []

    while generated_count < num_sequences and iteration_count < max_iterations:
        iteration_count += 1
        current_batch_size = min(batch_size, num_sequences - generated_count)

        generated = protgpt2(
            "<|endoftext|>",
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            num_return_sequences=current_batch_size,
            eos_token_id=0,
            pad_token_id=0
        )

        for item in generated:
            text = item['generated_text']

            text = text.replace("<|endoftext|>", "")

            text = re.sub(r'\s+', '', text)

            sequence = ''.join([c for c in text if c.isupper() and c.isalpha()])

            if sequence: 

                if len(sequence) >= sequence_length:
                    sequence = sequence[:sequence_length]
                    seq_len = len(sequence)
                    candidate_sequences.append((sequence, 0))  
                else:
                    seq_len = len(sequence)
                    if abs(seq_len - sequence_length) <= 10:
                        candidate_sequences.append((sequence, abs(seq_len - sequence_length)))

        if candidate_sequences:
            candidate_sequences.sort(key=lambda x: x[1])

            while candidate_sequences and generated_count < num_sequences:
                sequence, _ = candidate_sequences.pop(0)
                if sequence not in all_sequences:
                    all_sequences.append(sequence)
                    generated_count += 1
                    if generated_count % 10 == 0:
                        print(f"Generated Sequence: {generated_count}/{num_sequences}")
        elif iteration_count == 1:
            if generated:
                sample_text = generated[0]['generated_text'][:100]

        if generated_count < num_sequences and iteration_count >= max_iterations:
            if candidate_sequences:
                while candidate_sequences and generated_count < num_sequences:
                    sequence, _ = candidate_sequences.pop(0)
                    if sequence not in all_sequences:
                        all_sequences.append(sequence)
                        generated_count += 1
            break

    print(f"\n{len(all_sequences)} sequence generation completed")


    if all_sequences:
        df = pd.DataFrame({'Sequence': all_sequences})
        df.to_csv(output_path, index=False)
        print(f"{output_path} saved")
    else:
        print("Warning: generation x")
        df = pd.DataFrame({'Sequence': []})

    return df


def main():
    x = random.randint(1, 40)

    generated_df = inference(
        sequence_length=20,
        num_sequences=1,
        output_path="output.csv",
        seed=x,
        temperature=1.0
    )

    if not generated_df.empty:
        print("\nGenerated sequence:")
        for idx, seq in enumerate(generated_df['Sequence'], 1):
            print(f"{idx}. {seq}")
    else:
        print("\nno generation")

    print("\nOperation completed")


if __name__ == "__main__":
  main()