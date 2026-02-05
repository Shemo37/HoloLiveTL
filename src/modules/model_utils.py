
import os
import shutil
import requests
from tqdm import tqdm
import hashlib

def download_with_progress(url, destination, description="Downloading"):
    """Download file with progress bar"""
    if os.path.exists(destination):
        print(f"✅ {description} already exists at {destination}")
        return destination
    
    print(f"📥 {description} from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"✅ {description} completed")
        return destination
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        raise

def ensure_model_downloaded(model_id, model_cache_dir):
    """Download models if not present to avoid bundling them in the exe"""
    from huggingface_hub import snapshot_download
    
    model_dir = os.path.join(model_cache_dir, "whisper_model")
    vad_dir = os.path.join(model_cache_dir, "vad_model")
    
    # Check if model directory exists and has required files
    model_files_exist = False
    if os.path.exists(model_dir):
        required_files = ["model.safetensors", "config.json", "preprocessor_config.json"]
        model_files_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
    
    # Download Whisper model if needed
    if not model_files_exist:
        print("📥 Downloading Whisper model...")
        
        # Remove existing directory if incomplete
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(f"🗑️ Removed incomplete model directory: {model_dir}")
            except Exception as e:
                print(f"⚠️ Could not remove existing model directory: {e}")
        
        try:
            # Try using snapshot_download for a complete model download
            print("📥 Using Hugging Face snapshot download...")
            model_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=model_cache_dir,
                local_dir=model_dir,
                resume_download=True
            )
            print(f"✅ Model downloaded to: {model_dir}")
        except Exception as e:
            print(f"⚠️ Snapshot download failed: {e}")
            print("📥 Falling back to manual download...")
            
            # Fallback to manual download
            os.makedirs(model_dir, exist_ok=True)
            
            # Download essential files
            base_url = f"https://huggingface.co/{model_id}/resolve/main"
            
            files_to_download = [
                ("model.safetensors", "model.safetensors"),
                ("config.json", "config.json"),
                ("preprocessor_config.json", "preprocessor_config.json"),
                ("tokenizer.json", "tokenizer.json"),
                ("tokenizer_config.json", "tokenizer_config.json"),
                ("generation_config.json", "generation_config.json"),
                ("special_tokens_map.json", "special_tokens_map.json"),
                ("vocab.json", "vocab.json")
            ]
            
            for filename, local_name in files_to_download:
                url = f"{base_url}/{filename}"
                destination = os.path.join(model_dir, local_name)
                download_with_progress(url, destination, f"Downloading {filename}")
    
    # Download VAD model
    vad_path = os.path.join(vad_dir, "silero_vad.jit")
    if not os.path.exists(vad_path):
        print("📥 Downloading VAD model...")
        os.makedirs(vad_dir, exist_ok=True)
        
        # Try multiple URLs for the VAD model
        vad_urls = [
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.jit",
            "https://github.com/snakers4/silero-vad/raw/main/src/silero_vad/data/silero_vad.jit",
            "https://models.silero.ai/models/vad/silero_vad.jit",
            "https://github.com/snakers4/silero-vad/releases/download/v3.1/silero_vad.jit"
        ]
        
        vad_downloaded = False
        for vad_url in vad_urls:
            try:
                print(f"📥 Trying VAD model URL: {vad_url}")
                download_with_progress(vad_url, vad_path, "Downloading VAD model")
                vad_downloaded = True
                break
            except Exception as e:
                print(f"⚠️ Failed to download from {vad_url}: {e}")
                if os.path.exists(vad_path):
                    os.remove(vad_path)
                continue
        
        if not vad_downloaded:
            print("⚠️ Could not download VAD model from any source. VAD filtering will be disabled.")
            # Create a dummy file to prevent repeated download attempts
            with open(vad_path + ".failed", "w") as f:
                f.write("VAD model download failed")
    
    return model_dir, vad_dir

def get_file_hash(filepath):
    """Get SHA256 hash of a file"""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_kotoba_generate_kwargs(task="translate", target_language="en"):
    """Get appropriate generate_kwargs for kotoba-whisper-bilingual"""
    return {
        "language": target_language,
        "task": task,
        "temperature": 0.08,
        "max_new_tokens": 224,
        "no_repeat_ngram_size": 3,
        "suppress_tokens": [-1],
    }

def get_kotoba_pipeline_kwargs():
    """Pipeline-level configuration for kotoba-whisper"""
    return {
        "chunk_length_s": 15,
        "batch_size": 16,
        "return_timestamps": True,
    }

def optimize_for_vtuber_content(generate_kwargs):
    """Apply VTuber-specific optimizations to generate_kwargs"""
    generate_kwargs["temperature"] = 0.1
    generate_kwargs["no_repeat_ngram_size"] = 1
    return generate_kwargs
