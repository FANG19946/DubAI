# speaker_utils.py
from pydub import AudioSegment
from parse_srt import parse_srt
from dataclasses import dataclass, field
import torch
import torchaudio
import numpy as np
import io
from speechbrain.inference import SpeakerRecognition
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from typing import Optional, List, Set
import os
from collections import deque

os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"



@dataclass
class DialogueSegment:
    index: int
    start: int  # in ms
    end: int    # in ms
    audio: AudioSegment
    embedding: Optional[np.ndarray] = None
    temporal_neighbors: List[int] = field(default_factory=list)
    # These 2 fields really are bad practices, scheduled for CLEAN UP
    collected_indices: List[int] = field(default_factory=list)
    collected_audio: Optional[AudioSegment] = None

def gen_dialogue_array(audio_path, srt_path = 'eng_10_to_15.srt', temporal_span = 20000 ):
    
    base_wav_path = "data/wav/"  
    base_srt_path = "data/srt/"  

    audio = AudioSegment.from_wav(base_wav_path + audio_path)
    subtitles = parse_srt(base_srt_path + srt_path)
    
    segments = []
    

    # If any error suddenly arises on multi video diarization this is the part to check for BUG  
    for i,sub in enumerate(subtitles):
        start = sub["start"]
        end = sub["end"]
        segment = audio[start:end]
        embedding = extract_embedding(segment)
        segments.append(DialogueSegment(i+1, start, end, segment, embedding))

    for i, seg in enumerate(segments):
        
        for j in range(i - 1, -1, -1):
            if segments[j].end >= seg.start - temporal_span:
                seg.temporal_neighbors.append(j+1)
            else:
                break
        for j in range(i + 1, len(segments)):
            if segments[j].start <= seg.end + temporal_span:
                seg.temporal_neighbors.append(j+1)
            else:
                break
    

    for seg in segments:
        seg.collected_audio, seg.collected_indices = expand_speaker_segments(
            seed_seg=seg,
            segments=segments,
            k=2,
            alpha=0.7,
            max_audio_ms=20000
        )
    
    return segments
        


# Load the model once (move outside loop if calling multiple times)
embedding_model = SpeakerRecognition.from_hparams(
source="speechbrain/spkrec-ecapa-voxceleb",
savedir="pretrained_models/spkrec-ecapa-voxceleb",
run_opts={"local_file_strategy": "copy"},
)

def extract_embedding(audio_segment: AudioSegment):
    # Convert PyDub AudioSegment to raw bytes
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)

    # Normalize (optional but helps if input wasn't already normalized)
    samples /= np.iinfo(audio_segment.array_type).max

    # Convert to mono if stereo
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
        samples = samples.mean(axis=1)

    # Convert to torch tensor with shape [1, time]
    waveform = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)

   


    # Get embedding
    embedding = embedding_model.encode_batch(waveform)

    # Shape [1, 1, 192] -> squeeze to [192]
    return embedding.squeeze().detach().cpu().numpy()



def plot_speaker_embeddings(embeddings, save_name="umap_clusters.png", title="Speaker Embedding Clusters via UMAP"):
    base_plot_path = "data/plots/"

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(12, 8), dpi=150)

    # Plot each point as a number (1-based index)
    for i, (x, y) in enumerate(embedding_2d):
        plt.text(x, y, str(i + 1), fontsize=50, ha='center', va='center', color='mediumslateblue', weight='bold')

    plt.title(title, fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(True, linestyle='--', alpha=0.4)

    full_path = base_plot_path + save_name
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()




def update_embedding_anchor(
    embedding_anchor: np.ndarray,
    new_embedding: np.ndarray,
    alpha: float = 0.7
) -> np.ndarray:
    """
    Updates the speaker embedding anchor using a new segment embedding.

    Args:
        (X, Y, alpha) -> alpha * X + ( 1 - alpha ) * Y  
        (default 0.7)

    Returns:
        Normalized updated embedding anchor
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    combined = alpha * embedding_anchor + (1 - alpha) * new_embedding
    return combined / np.linalg.norm(combined)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_k_similar_neighbors(
    seg: DialogueSegment,
    segments: List[DialogueSegment],
    base_embedding: np.ndarray,
    visited: set[int],
    alpha: float = 0.7,
    k: int = 2,
) -> List[int]:
    """
    Returns top-k most similar temporal neighbors of `seg` using a weighted embedding anchor.

    Args:
        seg: The DialogueSegment for which neighbors are being selected.
        segments: List of all DialogueSegments.
        base_embedding: The current reference embedding to update with seg.embedding.
        k: Number of top neighbors to return.
        alpha: Weight for embedding_anchor (how much to bias towards base_embedding).

    Returns:
        List of neighbor indices (1-based, SRT indexing) sorted by similarity descending.
    """
    if seg.embedding is None:
        return []

    # Compute the updated anchor once
    updated_anchor = update_embedding_anchor(base_embedding, seg.embedding, alpha)

    scored_neighbors = []
    for neighbor_idx in seg.temporal_neighbors:
        
        if neighbor_idx in visited:
            continue

        neighbor_seg = segments[neighbor_idx - 1]  # convert 1-based to 0-based
        if neighbor_seg.embedding is None:
            continue
        sim = cosine_similarity(updated_anchor, neighbor_seg.embedding)
        scored_neighbors.append((neighbor_idx, sim))

    # Sort by similarity descending
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)

    # Return only the indices of top-k neighbors
    return [idx for idx, _ in scored_neighbors[:k]]


def expand_speaker_segments(
    seed_seg: DialogueSegment,
    segments: list[DialogueSegment],
    k: int = 2,
    alpha: float = 0.7,
    max_audio_ms: int = 20000
) -> AudioSegment:
    """
    Expands speaker audio starting from a seed DialogueSegment
    using top-k similar temporal neighbors.

    Returns:
        AudioSegment containing collected speaker audio
    """

    if seed_seg.embedding is None:
        return AudioSegment.silent(duration=0)

    # Collected audio
    collected_audio = seed_seg.audio
    collected_duration = len(collected_audio)
    collected_indices = [seed_seg.index]

    # Traversal state
    visited = set([seed_seg.index])  # 1-based indices
    queue = deque([seed_seg])

    # Anchor embedding starts as seed
    anchor_embedding = seed_seg.embedding.copy()

    while queue and collected_duration < max_audio_ms:
        current_seg = queue.popleft()

        # Get top-k unvisited neighbors
        neighbors = get_k_similar_neighbors(
            seg=current_seg,
            segments=segments,
            base_embedding=anchor_embedding,
            alpha=alpha,
            k=k,
            visited=visited
        )

        if not neighbors:
            continue  # dead-end for this path

        for neighbor_idx in neighbors:
            if collected_duration >= max_audio_ms:
                break

            visited.add(neighbor_idx)
            neighbor_seg = segments[neighbor_idx - 1]

            # Add audio
            collected_audio += neighbor_seg.audio
            collected_duration += len(neighbor_seg.audio)
            collected_indices.append(neighbor_idx)

            # Update anchor embedding
            anchor_embedding = update_embedding_anchor(
                anchor_embedding,
                neighbor_seg.embedding,
                alpha
            )

            queue.append(neighbor_seg)

    return collected_audio, collected_indices


def log_all_segments(segments, top_k=2):
    """
    Logs each segment's index and its top-k neighbors based on similarity.
    """
    for seg in segments:
        # Pair neighbors with similarities
        neighbor_pairs = list(zip(seg.temporal_neighbors, seg.neighbor_similarity))
        # Sort neighbors by their SRT index (ascending) instead of similarity
        neighbor_pairs.sort(key=lambda x: x[0])  

        # Take top-k
        top_neighbors = neighbor_pairs[:top_k]

        print(f"Segment {seg.index} | Top Neighbors: ", end="")
        for j, sim in top_neighbors:
            print(f"{j}:{sim:.3f}", end="  ")
        print()  # newline



if __name__ == "__main__":
    import os
    import json
    import random

    OUTPUT_DIR = "data/collected_audio_test"
    AUDIO_OUT_DIR = os.path.join(OUTPUT_DIR, "audio")

    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

    segments = gen_dialogue_array(
        audio_path="eng_10_to_15.wav",
        srt_path="eng_10_to_15.srt",
        temporal_span=20000
    )

    # Randomly pick up to 10 segments to save audio for
    save_segments = random.sample(
        segments,
        k=min(10, len(segments))
    )
    save_indices = {seg.index for seg in save_segments}

    print(f"Total segments: {len(segments)}")

    # Optional debugging
    # log_all_segments(segments, 4)

    metadata_segments = []

    for seg in segments:
        if seg.index in save_indices:
            wav_filename = f"{seg.index}.wav"
            wav_path = os.path.join(AUDIO_OUT_DIR, wav_filename)

            seg.collected_audio.export(wav_path, format="wav")
            audio_path = f"audio/{wav_filename}"
        else:
            audio_path = None

        metadata_segments.append({
            "id": seg.index,
            "collected_indices": seg.collected_indices})


    metadata = {
        "version": "v1",
        "audio_source": "eng_10_to_15.wav",
        "srt_source": "eng_10_to_15.srt",
        "params": {
            "temporal_span": 20000,
            "k": 2,
            "alpha": 0.7,
            "max_audio_ms": 20000
        },
        "segments": metadata_segments
    }

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved collected audio + metadata to: {OUTPUT_DIR}")
