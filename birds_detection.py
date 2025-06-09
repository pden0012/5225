#!/usr/bin/env python
# coding: utf-8

# requirements
# !pip install ultralytics supervision

from ultralytics import YOLO
import supervision as sv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import requests
import cv2

def image_prediction(image_path, result_filename=None, save_dir = "./image_prediction_results", confidence=0.5, model="./model.pt"):
    """
    Function to display predictions of a pre-trained YOLO model on a given image.

    Parameters:
        image_path (str): Path to the image file. Can be a local path or a URL.
        result_path (str): If not None, this is the output filename.
        confidence (float): 0-1, only results over this value are saved.
        model (str): path to the model.
    """

    # Load YOLO model
    model = YOLO(model)
    class_dict = model.names

    # Load image from local path
    img = cv.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print("Couldn't load the image! Please check the image path.")
        return

    # Get image dimensions
    h, w = img.shape[:2]

    # Calculate optimal thickness for boxes and text based on image resolution
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))

    # Set up color palette for annotations
    color_palette = sv.ColorPalette.from_matplotlib('magma', 10)

    # Create box and label annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness, color=color_palette)
    label_annotator = sv.LabelAnnotator(color=color_palette, text_scale=text_scale, 
                                        text_thickness=thickness, 
                                        text_position=sv.Position.TOP_LEFT)

    # Run the model on the image
    result = model(img)[0]

    # Convert YOLO result to Detections format
    detections = sv.Detections.from_ultralytics(result)
    bird_tags = [] 
    labels = []  
    
    # Filter detections based on confidence threshold and check if any exist
    if detections.class_id is not None:
        detections = detections[(detections.confidence > confidence)]
        
        if len(detections) > 0: 
            bird_tags = [class_dict[cls_id] for cls_id in detections.class_id]
            print(f"Detected birds: {bird_tags}")  # Added log output
            
            # Create labels for the detected objects
            labels = [f"{class_dict[cls_id]} {conf*100:.2f}%" for cls_id, conf in 
                     zip(detections.class_id, detections.confidence)]

        # Annotate the image with boxes and labels
        box_annotator.annotate(img, detections=detections)
        label_annotator.annotate(img, detections=detections, labels=labels)

    if result_filename:
        os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
        save_path = os.path.join(save_dir, result_filename)
        try:
            status = cv.imwrite(save_path, img)
            print(f"Image save status = {status}.")
        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        print("Filename is none, result is not saved.")

    return bird_tags 

# ## Video Detection
def video_prediction(
    video_path, 
    result_filename=None, 
    save_dir="./video_prediction_results", 
    confidence=0.5, 
    model="./model.pt", 
    sample_interval=5
):
    """
    🔥 New logic: Support complete saving of multiple bird species + confidence-based selection for single species
    """
    # 🔥 Record maximum count per frame for each bird species and corresponding information
    max_count_per_bird = {}  # {species: {'max_count': int, 'confidence_at_max': float, 'frame_at_max': int}}
    all_detections = []  # Used to calculate single-frame totals
    frame_detections = {}  # {frame_num: [detections]} Used to calculate single-frame totals
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    yolo_model = YOLO(model)
    class_dict = yolo_model.names

    if result_filename:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, result_filename)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps // sample_interval), (int(w), int(h)))
    else:
        out = None

    processed_frames = 0

    for frame_num in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        result = yolo_model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[(detections.confidence > confidence)]

        # 🔥 Count each bird species in current frame and their confidence
        current_frame_detections = []
        frame_bird_counts = {}
        frame_bird_confidences = {}
        
        for cls_id, conf in zip(detections.class_id, detections.confidence):
            bird_name = class_dict[cls_id]
            
            # Record individual detection
            detection_info = {
                'bird_name': bird_name,
                'confidence': float(conf),
                'frame_num': frame_num
            }
            all_detections.append(detection_info)
            current_frame_detections.append(detection_info)
            
            # Count in current frame
            frame_bird_counts[bird_name] = frame_bird_counts.get(bird_name, 0) + 1
            
            # Record all confidence values for this bird species in current frame
            if bird_name not in frame_bird_confidences:
                frame_bird_confidences[bird_name] = []
            frame_bird_confidences[bird_name].append(float(conf))

        # 🔥 Save current frame detection results (grouped by bird species)
        if frame_bird_counts:
            frame_detections[frame_num] = current_frame_detections
        
        # 🔥 Update maximum single-frame count record for each bird species
        for bird_name, count in frame_bird_counts.items():
            # Choose the highest confidence for this bird species in current frame as representative
            representative_confidence = max(frame_bird_confidences[bird_name])
            
            if bird_name not in max_count_per_bird or count > max_count_per_bird[bird_name]['max_count']:
                max_count_per_bird[bird_name] = {
                    'max_count': count,
                    'confidence_at_max': representative_confidence,
                    'frame_at_max': frame_num
                }

        # Display current frame detection results
        if frame_bird_counts:
            frame_summary = []
            for bird, count in frame_bird_counts.items():
                max_conf = max(frame_bird_confidences[bird])
                frame_summary.append(f"{bird}×{count}({max_conf*100:.1f}%)")
            print(f"Frame {frame_num}: {', '.join(frame_summary)}")

        if out:
            out.write(frame)
        processed_frames += 1

    cap.release()
    if out:
        out.release()

    # 🔥 Calculate bird distribution per frame (to find frame with most total birds)
    frame_bird_distribution = {}  # {frame_num: {species: count}}
    for frame_num, detections_list in frame_detections.items():
        frame_birds = {}
        for detection in detections_list:
            bird_name = detection['bird_name']
            frame_birds[bird_name] = frame_birds.get(bird_name, 0) + 1
        frame_bird_distribution[frame_num] = frame_birds
    
    # 🔥 Calculate frame with most total birds
    frame_totals = {}
    for frame_num, detections_list in frame_detections.items():
        frame_totals[frame_num] = len(detections_list)
    
    best_total_frame = None
    if frame_totals:
        best_total_frame = max(frame_totals.items(), key=lambda x: x[1])

    print(f"Total video frames: {frame_count}")
    print(f"Actually processed frames: {processed_frames}")
    print(f"Total detections: {len(all_detections)}")
    print(f"Number of bird species detected: {len(max_count_per_bird)}")
    
    # 🔥 Display maximum single-frame count for each bird species
    print("\n📊 Maximum single-frame count for each bird species:")
    for bird, info in max_count_per_bird.items():
        print(f"  {bird}: Max {info['max_count']} birds (confidence {info['confidence_at_max']*100:.2f}% - Frame {info['frame_at_max']})")
    
    if best_total_frame:
        print(f"\n🎬 Frame with most total birds: Frame {best_total_frame[0]} ({best_total_frame[1]} birds)")

    return {
        "max_count_per_bird": max_count_per_bird,
        "all_detections": all_detections,
        "frame_detections": frame_detections,
        "frame_bird_distribution": frame_bird_distribution,  # 🔥 New: Bird distribution per frame
        "frame_totals": frame_totals,  # 🔥 Total birds per frame
        "best_total_frame": best_total_frame,  # 🔥 Frame with most total birds
        "total_frames": frame_count,
        "processed_frames": processed_frames,
        "total_detections": len(all_detections),
        "species_count": len(max_count_per_bird)  # 🔥 Number of species
    }

if __name__ == '__main__':
    print("predicting...")
    image_prediction("./test_images/crows_1.jpg", result_filename="crows_result1.jpg")
    image_prediction("./test_images/crows_3.jpg", result_filename='crows_detected_2.jpg')
    image_prediction("./test_images/kingfisher_2.jpg",result_filename='kingfishers_detected.jpg' )
    image_prediction("./test_images/myna_1.jpg",result_filename='myna_detected.jpg')
    image_prediction("./test_images/owl_2.jpg",result_filename='owls_detected.jpg')
    image_prediction("./test_images/peacocks_3.jpg",result_filename='peacocks_detected_1.jpg')
    image_prediction('./test_images/sparrow_3.jpg',result_filename='sparrow_detected_1.jpg')
    image_prediction('./test_images/sparrow_1.jpg',result_filename='sparrow_detected_2.jpg')

    # uncomment to test video prediction
    # video_prediction("./test_videos/crows.mp4",result_filename='crows_detected.mp4')
    # video_prediction("./test_videos/kingfisher.mp4",result_filename='kingfisher_detected.mp4')

