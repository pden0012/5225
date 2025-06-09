#!/usr/bin/env python3

# 🔥 Step 1: Import os and set environment variables immediately (before all other imports)
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_DISABLE_CACHING"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "safe"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["XDG_CACHE_HOME"] = "/tmp"
os.environ["MPLCONFIGDIR"] = "/tmp/.matplotlib"
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"

# 🔥 Step 2: Before any import numba, inject our own "fake numba" module
import types

# Construct numba stub
fake_numba = types.ModuleType("numba")

# 🔥 Key: Add all necessary decorators (missing in original code!)
fake_numba.jit = lambda *args, **kwargs: (lambda fn: fn)
fake_numba.njit = lambda *args, **kwargs: (lambda fn: fn)
fake_numba.vectorize = lambda *args, **kwargs: (lambda fn: fn)  # ⭐ Missing
fake_numba.guvectorize = lambda *args, **kwargs: (lambda fn: fn)  # ⭐ Critical missing!
fake_numba.stencil = lambda *args, **kwargs: (lambda fn: fn)  # ⭐ Missing
fake_numba.jitclass = lambda *args, **kwargs: (lambda fn: fn)  # ⭐ Missing
fake_numba.cfunc = lambda *args, **kwargs: (lambda fn: fn)  # ⭐ Missing
fake_numba.prange = range  # ⭐ Add this line: prange replaced with regular range

# config needs to provide CACHE_DIR, DISABLE_JIT, THREADING_LAYER
fake_numba.config = types.SimpleNamespace(
    CACHE_DIR=os.environ["NUMBA_CACHE_DIR"],
    DISABLE_JIT=True,
    THREADING_LAYER=os.environ["NUMBA_THREADING_LAYER"]
)

# 🔥 Key: Create more necessary submodules (missing in original code!)
fake_numba.core = types.ModuleType("numba.core")
fake_numba.core.config = fake_numba.config

# Create numba.np submodule system
fake_numba.np = types.ModuleType("numba.np")
fake_numba.np.ufunc = types.ModuleType("numba.np.ufunc")
fake_numba.np.ufunc.decorators = types.ModuleType("numba.np.ufunc.decorators")

# Add decorators to submodules as well
fake_numba.np.ufunc.decorators.vectorize = fake_numba.vectorize
fake_numba.np.ufunc.decorators.guvectorize = fake_numba.guvectorize  # ⭐ Key!

# Inject into sys.modules
sys.modules["numba"] = fake_numba
sys.modules["numba.config"] = fake_numba.config
sys.modules["numba.core"] = fake_numba.core
sys.modules["numba.core.config"] = fake_numba.config

# ⭐ Key: Add missing submodule injection from original code
sys.modules["numba.np"] = fake_numba.np
sys.modules["numba.np.ufunc"] = fake_numba.np.ufunc
sys.modules["numba.np.ufunc.decorators"] = fake_numba.np.ufunc.decorators

# Other potentially needed submodules
additional_modules = [
    "numba.core.decorators",
    "numba.core.types", 
    "numba.typed",
    "numba.stencils",
    "numba.stencils.stencil"
]

for module_name in additional_modules:
    sys.modules[module_name] = fake_numba

print("✅ Complete fake numba module injected")

# 🔥 Normal import and initialization process begins below

# Verify environment variable settings
print("🔍 Environment variable verification:")
print(f"  NUMBA_DISABLE_JIT: {os.environ.get('NUMBA_DISABLE_JIT')}")
print(f"  NUMBA_DISABLE_CACHING: {os.environ.get('NUMBA_DISABLE_CACHING')}")

# Create necessary cache directories
os.makedirs("/tmp/numba_cache", exist_ok=True)
os.makedirs("/tmp/.matplotlib", exist_ok=True)

# 🔥 Step 3: Import all modules (one-time import, avoid repetition)
import json
import io
import traceback
import subprocess
import tempfile
from collections import Counter
from urllib.parse import unquote_plus
import csv
import numpy as np
from contextlib import redirect_stderr

# 🔥 Key: Test librosa import (missing test in original code!)
print("🔍 Testing librosa import...")
try:
    import librosa
    print("✅ librosa import successful")
    
    # Simple test of librosa functionality
    test_signal = np.array([0.1, 0.2, 0.1], dtype=np.float32)
    import soundfile as sf
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, test_signal, 22050, format='WAV')
        sig, sr = librosa.load(tmp_file.name, sr=6000, mono=True)
        print(f"✅ librosa.load test successful: {len(sig)} samples at {sr} Hz")
        os.unlink(tmp_file.name)
        
except Exception as e:
    print(f"❌ librosa import/test failed: {e}")
    traceback.print_exc()

# 🔥 Step 4: Monkey-patch tflite_runtime.interpreter
import tensorflow.lite as _tf_lite 
sys.modules['tflite_runtime.interpreter'] = _tf_lite

# 🔥 Step 5: Configure boto3, cv2, etc.
import boto3
import cv2

# 🔥 Step 6: Add your project path and import dependencies
sys.path.append("./BirdNET-Analyzer")

print("🔍 Preparing to import birdnet_analyzer...")
try:
    from birdnet_analyzer.analyze.core import analyze
    print("✅ birdnet_analyzer.analyze.core import successful")
except Exception as e:
    print(f"❌ birdnet_analyzer.analyze.core import failed: {e}")
    traceback.print_exc()

try:
    from birds_detection import image_prediction, video_prediction
    print("✅ birds_detection import successful")
except Exception as e:
    print(f"❌ birds_detection import failed: {e}")

# DynamoDB / S3 clients
dynamodb = boto3.resource("dynamodb")
table    = dynamodb.Table("FilesMetadata")
s3       = boto3.client("s3")

def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key    = unquote_plus(record["s3"]["object"]["key"])   # e.g. "bird_input/foo.jpg" or "bird_input/bar.mp4"

        # Only process objects in bird_input/ directory
        if not key.startswith("bird_input/"):
            continue

        # 1) Download to /tmp
        basename   = os.path.basename(key)          # e.g. "foo.jpg" or "bar.mp4"
        local_path = f"/tmp/{basename}"              # "/tmp/foo.jpg" or "/tmp/bar.mp4"
        try:
            s3.download_file(bucket, key, local_path)
        except Exception as e:
            print(f"ERROR: Download failed {key}: {e}")
            continue

        # 2) Infer file type from extension
        file_type = basename.split(".")[-1].lower()   # "jpg"/"png"/"mp4"/"avi"
        tag_counts = Counter()

        # 3) Determine if image or video, then call appropriate function
        if file_type in ["jpg", "jpeg", "png"]:
            # ——Image branch: call image_prediction()——
            result_fname   = f"result_{basename}"       # e.g. "result_foo.jpg"
            full_temp_path = f"/tmp/{result_fname}"

            # Capture image_prediction print logs by temporarily redirecting stdout
            old_stdout = sys.stdout
            sys.stdout  = io.StringIO()

            try:
                bird_list = image_prediction(
                    image_path    = local_path,
                    result_filename = result_fname,
                    save_dir      = "/tmp",
                    confidence    = 0.5
                ) or []
            except Exception as e:
                print(f"ERROR: image_prediction call failed: {e}")
                bird_list = []
            finally:
                captured = sys.stdout.getvalue()
                sys.stdout = old_stdout
                print("DEBUG: image_prediction logs:\n", captured)

            # 🔥 Modified: Count and select only the most numerous species
            if isinstance(bird_list, list) and bird_list:
                # Count each species
                tag_counts = Counter([b.lower() for b in bird_list])
                
                # Find the most numerous species
                max_species = max(tag_counts, key=tag_counts.get) 
                max_count = tag_counts[max_species]
                bird_tags = {max_species: max_count}
                
                print(f"🔍 Complete detection results: {dict(tag_counts)}")
                print(f"🐦 Most numerous bird species in image: {max_species} ({max_count} birds)")
            else:
                bird_tags = {}

            # Upload detection image to detections/
            if os.path.exists(full_temp_path):
                detection_key = key.replace("bird_input/", "detections/")
                try:
                    s3.upload_file(full_temp_path, bucket, detection_key)
                    print(f"Uploaded detection result image to s3://{bucket}/{detection_key}")
                except Exception as e:
                    print(f"ERROR: Upload detection result image failed {full_temp_path}: {e}")
                finally:
                    try:
                        os.remove(full_temp_path)
                    except:
                        pass

            # Save image detection results to DynamoDB
            item = {
                "id": context.aws_request_id,
                "file_type": file_type,
                "bird_tags": bird_tags,  # Now only contains the most numerous one
                "s3_url": f"s3://{bucket}/{key}",
            }
            thumb_key = key.replace("bird_input/", "thumbnails/")
            item["thumbnail_url"] = f"s3://{bucket}/{thumb_key}"
            
            print(f"🔍 Preparing to write to DynamoDB: {item}")
            
            # Write to DynamoDB
            try:
                table.put_item(Item=item)
                print("✅ Image written to DynamoDB successfully:", item)
            except Exception as e:
                print("❌ Image write to DynamoDB failed:", e)
                traceback.print_exc()

        elif file_type in ["mp4", "avi", "mov", "mkv"]:
            # ——Video branch: call video_prediction()——
            result_fname   = f"result_{basename}"       # e.g. "result_bar.mp4"
            full_temp_path = f"/tmp/{result_fname}"

            old_stdout = sys.stdout
            sys.stdout  = io.StringIO()

            try:
                result = video_prediction(
                    video_path     = local_path,
                    result_filename = result_fname,
                    save_dir       = "/tmp",
                    confidence     = 0.5,
                    model          = "./model.pt",
                    sample_interval= 3
                ) or {}
            except Exception as e:
                print(f"ERROR: video_prediction call failed: {e}")
                result = {}
            finally:
                captured = sys.stdout.getvalue()
                sys.stdout = old_stdout
                print("DEBUG: video_prediction logs:\n", captured)

            # 🔥 Correct video processing logic (as required by teacher)
            frame_detections = result.get("frame_detections", {})

            if frame_detections:
                # Calculate total bird count and bird distribution per frame
                frame_bird_counts = {}  # {frame_num: {species: count}}
                frame_totals = {}       # {frame_num: total_count}
                
                for frame_num, detections_list in frame_detections.items():
                    frame_birds = {}
                    for detection in detections_list:
                        bird_name = detection['bird_name']
                        frame_birds[bird_name] = frame_birds.get(bird_name, 0) + 1
                    
                    frame_bird_counts[frame_num] = frame_birds
                    frame_totals[frame_num] = sum(frame_birds.values())
                
                # Find the frame with the most total birds
                if frame_totals:
                    best_frame_num = max(frame_totals, key=frame_totals.get)
                    best_frame_birds = frame_bird_counts[best_frame_num]
                    
                    # Save all birds from that frame
                    bird_tags = best_frame_birds.copy()
                    
                    print(f"🏆 Selected frame: Frame {best_frame_num} (total {frame_totals[best_frame_num]} birds)")
                    print(f"📊 Bird distribution in this frame: {bird_tags}")

            else:
                bird_tags = {}
                print("❌ No birds detected in video")

            # Upload detection video
            if os.path.exists(full_temp_path):
                detection_key = key.replace("bird_input/", "detections/")
                try:
                    s3.upload_file(full_temp_path, bucket, detection_key)
                    print(f"Uploaded detection result video to s3://{bucket}/{detection_key}")
                except Exception as e:
                    print(f"ERROR: Upload detection result video failed {full_temp_path}: {e}")
                finally:
                    try:
                        os.remove(full_temp_path)
                    except:
                        pass
            # Prepare DynamoDB record
            item = {
                "id": context.aws_request_id,
                "file_type": file_type,
                "s3_url": f"s3://{bucket}/{key}",
                "bird_tags": bird_tags  # Now contains confidence percentage values
            }
            
            # 🔥 Write to DynamoDB
            try:
                table.put_item(Item=item)
                print("✅ Video written to DynamoDB:", item)
            except Exception as e:
                print("❌ Video write to DynamoDB failed:", e)

        elif file_type in ["wav", "mp3", "flac"]:
            print(f"File exists: {os.path.exists(local_path)}, size: {os.path.getsize(local_path) if os.path.exists(local_path) else 'N/A'}")

            # 1) Resample to standard WAV
            converted = "/tmp/converted.wav"
            try:
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", local_path,
                    "-ac", "1",           # mono
                    "-ar", "48000",       # BirdNET standard sample rate
                    "-sample_fmt", "s16", # 16-bit
                    converted
                ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print("✅ ffmpeg conversion successful:", converted)
                target = converted
            except Exception as e:
                print("❌ ffmpeg conversion failed, using original file:", e)
                target = local_path

            # 2) 🔥 Suppress warnings during audio analysis
            try:
                print("🎵 Starting BirdNET audio analysis...")
                
                # Temporarily redirect stderr to suppress TensorFlow warnings
                stderr_buffer = io.StringIO()
                
                with redirect_stderr(stderr_buffer):
                    result = analyze(
                        input=target,
                        output=None,
                        min_conf=0.5,
                        rtype="csv",
                        threads=1
                    )
                
                # Only show important error messages (if any)
                stderr_content = stderr_buffer.getvalue()
                if "ERROR" in stderr_content or "FATAL" in stderr_content:
                    print("⚠️ Found important error messages:")
                    print(stderr_content)
                
                print("✅ analyze returned:", result)
            except Exception as e:
                print("❌ analyze threw exception:", e)
                traceback.print_exc()
                result = None

            # 3) Extract species tags - audio files
            bird_species = {}
            if isinstance(result, dict) and result.get('status') == 'success':
                species_data = result.get('species')
                
                if isinstance(species_data, dict):
                    # Single species detection result
                    name = species_data.get('name') or species_data.get('species')
                    confidence = species_data.get('confidence', 0)
                    if name:
                        bird_species[name] = 1  # If detected, save as 1
                        print(f"🐦 Bird detected: {name} (confidence: {confidence:.2%}) -> saved as 1")
                
                elif isinstance(species_data, list):
                    # Multiple species detection results - choose highest confidence
                    if species_data:
                        # Sort by confidence, choose highest
                        best_detection = max(species_data, key=lambda x: x.get('confidence', 0))
                        name = best_detection.get('name') or best_detection.get('species')
                        confidence = best_detection.get('confidence', 0)
                        if name:
                            bird_species[name] = 1
                            print(f"🐦 Bird detected: {name} (confidence: {confidence:.2%}) -> saved as 1")
                            print(f"🔍 All detection results: {[(s.get('name', 'Unknown'), s.get('confidence', 0)) for s in species_data]}")

            print(f"🔍 Audio detection results: {bird_species}")

            # 4) Write to DynamoDB
            item = {
                'id': context.aws_request_id,
                'file_type': file_type,
                's3_url': f"s3://{bucket}/{key}",
                'bird_tags': bird_species
            }
            try:
                table.put_item(Item=item)
                print("✅ Audio detection results written to DynamoDB:", item)
            except Exception as e:
                print("❌ Write to DynamoDB failed:", e)
                traceback.print_exc()

        else:
            continue

    return { 'statusCode': 200, 'body': json.dumps('SUCCESS') }

