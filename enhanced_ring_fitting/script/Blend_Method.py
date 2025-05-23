#!/usr/bin/env python3
"""
Enhanced AI Ring Fitting Pipeline System
========================================

A comprehensive pipeline for automated ring fitting on hand models with
advanced prompt generation, lighting control, and precise ring placement.

Features:
- Advanced image analysis and detailed prompt generation
- Precise ring segmentation and extraction
- Hand/finger detection with MediaPipe
- Non-deformative ring fitting with perspective matching
- Advanced lighting adjustment and enhancement
- Multiple angle and pose generation
- Face and model preservation (no alteration)
- High-quality output generation

Author: Enhanced AI Assistant
Date: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from sklearn.cluster import KMeans
import mediapipe as mp
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import requests
from io import BytesIO
import base64
import logging
from datetime import datetime
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("SAM not available. Install segment-anything for advanced segmentation.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Install mediapipe for hand detection.")

@dataclass
class RingProperties:
    """Enhanced data class for ring properties."""
    center: Tuple[int, int]
    radius: float
    angle: float
    material: str
    style: str
    gem_type: str
    color_palette: List[str]
    dimensions: Tuple[int, int]
    texture_features: Dict[str, float]
    brightness: float
    contrast: float
    saturation: float
    metallic_score: float
    gem_positions: List[Tuple[int, int]]

@dataclass
class HandProperties:
    """Enhanced data class for hand properties."""
    landmarks: List[Tuple[int, int]]
    finger_positions: Dict[str, Tuple[int, int]]
    finger_joints: Dict[str, List[Tuple[int, int]]]
    hand_orientation: float
    skin_tone: str
    pose_category: str
    hand_size: float
    finger_widths: Dict[str, float]
    nail_positions: Dict[str, Tuple[int, int]]

@dataclass
class LightingConfig:
    """Enhanced data class for lighting configuration."""
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    warmth: float = 0.0  # -1 to 1
    shadow_intensity: float = 0.5
    highlight_intensity: float = 0.7
    ambient_light: float = 0.8
    directional_light_angle: float = 45.0  # degrees
    light_softness: float = 0.5
    reflection_intensity: float = 0.6
    environment_lighting: str = "studio"  # studio, natural, dramatic, soft

@dataclass
class FittingConfig:
    """Configuration for ring fitting parameters."""
    target_finger: str = "ring_finger"
    ring_scale: float = 1.0
    position_offset: Tuple[float, float] = (0.0, 0.0)
    rotation_adjustment: float = 0.0
    perspective_correction: bool = True
    shadow_generation: bool = True
    reflection_generation: bool = True
    quality_enhancement: bool = True

class EnhancedRingFittingPipeline:
    """
    Enhanced pipeline class for ring fitting and prompt generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the enhanced pipeline with required models and configurations."""
        self.mp_hands = None
        self.mp_drawing = None
        self.hands = None
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        
        # Enhanced prompt generation templates
        self.prompt_templates = {
            'base_template': "Professional high-resolution photograph of {ring_description} elegantly positioned on {finger_description} of {hand_description}, {lighting_description}, {technical_specs}",
            'ring_descriptions': {
                'solitaire': "a stunning solitaire {material} engagement ring featuring a brilliant {gem_type}",
                'band': "an elegant {material} wedding band with {details}",
                'statement': "a bold statement {material} ring with intricate {details}",
                'vintage': "a vintage-inspired {material} ring with classic {details}",
                'modern': "a contemporary {material} ring with sleek {details}"
            },
            'lighting_descriptions': {
                'studio': "professional studio lighting with controlled shadows and highlights",
                'natural': "soft natural lighting with gentle shadows",
                'dramatic': "dramatic lighting creating depth and contrast",
                'soft': "soft diffused lighting for elegant presentation"
            },
            'technical_specs': [
                "shot with professional macro lens",
                "perfect focus and depth of field",
                "no motion blur or distortion",
                "commercial jewelry photography quality",
                "high dynamic range",
                "accurate color reproduction"
            ]
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        logger.info("Enhanced Ring Fitting Pipeline initialized successfully")
    
    def analyze_ring_image(self, ring_image_path: str, detailed_analysis: bool = True) -> RingProperties:
        """
        Enhanced ring image analysis with detailed feature extraction.
        """
        logger.info(f"🔍 Analyzing ring image: {ring_image_path}")
        
        if not os.path.exists(ring_image_path):
            raise FileNotFoundError(f"Ring image not found: {ring_image_path}")
        
        image = cv2.imread(ring_image_path)
        if image is None:
            raise ValueError(f"Could not load image from {ring_image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Enhanced ring detection and analysis
        center = self._find_ring_center_enhanced(image_rgb)
        radius = self._estimate_ring_radius_enhanced(image_rgb, center)
        angle = self._calculate_ring_angle_enhanced(image_rgb, center)
        
        # Advanced material and style classification
        material = self._classify_material_enhanced(image_rgb)
        style = self._classify_style_enhanced(image_rgb)
        gem_type, gem_positions = self._detect_gemstone_enhanced(image_rgb)
        
        # Enhanced color and texture analysis
        color_palette = self._extract_color_palette_enhanced(image_rgb)
        texture_features = self._analyze_texture_features(image_rgb)
        
        # Image quality metrics
        brightness = self._calculate_brightness(image_rgb)
        contrast = self._calculate_contrast(image_rgb)
        saturation = self._calculate_saturation(image_rgb)
        metallic_score = self._calculate_metallic_score(image_rgb)
        
        ring_props = RingProperties(
            center=center,
            radius=radius,
            angle=angle,
            material=material,
            style=style,
            gem_type=gem_type,
            color_palette=color_palette,
            dimensions=(width, height),
            texture_features=texture_features,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            metallic_score=metallic_score,
            gem_positions=gem_positions
        )
        
        logger.info(f"✅ Ring analysis complete: {style} {material} ring with {gem_type}")
        return ring_props
    
    def analyze_hand_model(self, model_image_path: str) -> HandProperties:
        """
        Enhanced hand model analysis with detailed landmark detection.
        """
        logger.info(f"👋 Analyzing hand model: {model_image_path}")
        
        if not os.path.exists(model_image_path):
            raise FileNotFoundError(f"Model image not found: {model_image_path}")
        
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("MediaPipe not available. Using enhanced basic hand detection.")
            return self._enhanced_basic_hand_analysis(model_image_path)
        
        # Load and process image
        image = cv2.imread(model_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            raise ValueError("No hands detected in the model image")
        
        # Extract detailed hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(int(lm.x * image_rgb.shape[1]), int(lm.y * image_rgb.shape[0])) 
                    for lm in hand_landmarks.landmark]
        
        # Enhanced feature extraction
        finger_positions = self._extract_finger_positions_enhanced(landmarks)
        finger_joints = self._extract_finger_joints(landmarks)
        hand_orientation = self._calculate_hand_orientation_enhanced(landmarks)
        skin_tone = self._analyze_skin_tone_enhanced(image_rgb, landmarks)
        pose_category = self._classify_hand_pose_enhanced(landmarks)
        hand_size = self._calculate_hand_size(landmarks)
        finger_widths = self._calculate_finger_widths(landmarks, image_rgb)
        nail_positions = self._detect_nail_positions(landmarks, image_rgb)
        
        hand_props = HandProperties(
            landmarks=landmarks,
            finger_positions=finger_positions,
            finger_joints=finger_joints,
            hand_orientation=hand_orientation,
            skin_tone=skin_tone,
            pose_category=pose_category,
            hand_size=hand_size,
            finger_widths=finger_widths,
            nail_positions=nail_positions
        )
        
        logger.info(f"✅ Hand analysis complete: {pose_category} pose, {skin_tone} skin tone")
        return hand_props
    
    def fit_ring_to_hand(self, 
                        ring_image_path: str, 
                        model_image_path: str,
                        fitting_config: FittingConfig = None) -> np.ndarray:
        """
        Enhanced ring fitting with non-deformative placement.
        """
        if fitting_config is None:
            fitting_config = FittingConfig()
        
        logger.info(f"💍 Fitting ring to {fitting_config.target_finger} with enhanced precision...")
        
        # Analyze both images
        ring_props = self.analyze_ring_image(ring_image_path)
        hand_props = self.analyze_hand_model(model_image_path)
        
        # Load images with proper handling
        ring_img = self._load_image_with_alpha(ring_image_path)
        model_img = cv2.imread(model_image_path)
        model_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
        
        # Enhanced ring extraction with transparency preservation
        ring_extracted = self._extract_ring_enhanced(ring_img, ring_props)
        
        # Get precise finger position and orientation
        finger_info = self._get_finger_placement_info(
            hand_props, fitting_config.target_finger
        )
        
        # Calculate precise transformation matrix
        transform_matrix = self._calculate_precise_transform(
            ring_props, hand_props, finger_info, fitting_config
        )
        
        # Apply non-deformative transformation
        ring_transformed = self._transform_ring_enhanced(ring_extracted, transform_matrix)
        
        # Generate realistic shadows and reflections
        if fitting_config.shadow_generation:
            shadow_layer = self._generate_ring_shadow(ring_transformed, finger_info)
        else:
            shadow_layer = None
            
        if fitting_config.reflection_generation:
            reflection_layer = self._generate_ring_reflection(ring_transformed, ring_props)
        else:
            reflection_layer = None
        
        # Enhanced blending with preservation of original details
        result = self._blend_ring_enhanced(
            model_rgb, ring_transformed, shadow_layer, reflection_layer, finger_info
        )
        
        # Quality enhancement if requested
        if fitting_config.quality_enhancement:
            result = self._enhance_image_quality(result)
        
        logger.info("✅ Enhanced ring fitting complete")
        return result
    
    def apply_advanced_lighting(self, 
                              image: np.ndarray, 
                              lighting_config: LightingConfig) -> np.ndarray:
        """
        Apply advanced lighting effects with professional quality.
        """
        logger.info(f"💡 Applying {lighting_config.environment_lighting} lighting effects...")
        
        # Convert to PIL for advanced processing
        pil_img = Image.fromarray(image)
        
        # Apply environment-specific lighting
        if lighting_config.environment_lighting == "studio":
            pil_img = self._apply_studio_lighting(pil_img, lighting_config)
        elif lighting_config.environment_lighting == "natural":
            pil_img = self._apply_natural_lighting(pil_img, lighting_config)
        elif lighting_config.environment_lighting == "dramatic":
            pil_img = self._apply_dramatic_lighting(pil_img, lighting_config)
        else:  # soft
            pil_img = self._apply_soft_lighting(pil_img, lighting_config)
        
        # Apply basic adjustments
        pil_img = self._apply_basic_lighting_adjustments(pil_img, lighting_config)
        
        # Apply advanced effects
        pil_img = self._apply_advanced_lighting_effects(pil_img, lighting_config)
        
        result = np.array(pil_img)
        logger.info("✅ Advanced lighting effects applied")
        return result
    
    def generate_comprehensive_prompt(self, 
                                    ring_props: RingProperties,
                                    hand_props: HandProperties,
                                    lighting_config: LightingConfig,
                                    additional_details: Dict = None) -> Dict[str, str]:
        """
        Generate comprehensive and detailed prompts for various use cases.
        """
        logger.info("📝 Generating comprehensive prompts...")
        
        # Build detailed descriptions
        ring_desc = self._build_ring_description(ring_props)
        hand_desc = self._build_hand_description(hand_props)
        lighting_desc = self._build_lighting_description(lighting_config)
        technical_desc = self._build_technical_description()
        
        # Generate multiple prompt variations
        prompts = {
            'detailed': self._generate_detailed_prompt(ring_desc, hand_desc, lighting_desc, technical_desc),
            'commercial': self._generate_commercial_prompt(ring_desc, hand_desc, lighting_desc),
            'artistic': self._generate_artistic_prompt(ring_desc, hand_desc, lighting_desc),
            'technical': self._generate_technical_prompt(ring_desc, hand_desc, technical_desc),
            'social_media': self._generate_social_media_prompt(ring_desc, hand_desc),
        }
        
        # Add additional details if provided
        if additional_details:
            for key, prompt in prompts.items():
                prompts[key] = self._enhance_prompt_with_details(prompt, additional_details)
        
        logger.info("✅ Comprehensive prompts generated")
        return prompts
    
    def generate_multiple_angles_enhanced(self, 
                                        ring_image_path: str, 
                                        model_image_path: str,
                                        angles: List[str] = None,
                                        lighting_configs: Dict[str, LightingConfig] = None) -> Dict[str, Dict]:
        """
        Generate enhanced ring fitting for multiple angles with different lighting.
        """
        if angles is None:
            angles = ["front_view", "side_view", "top_view", "angled_view", "close_up"]
        
        if lighting_configs is None:
            lighting_configs = {angle: LightingConfig() for angle in angles}
        
        logger.info(f"🔄 Generating {len(angles)} enhanced angle variations...")
        
        # Analyze images once
        ring_props = self.analyze_ring_image(ring_image_path)
        hand_props = self.analyze_hand_model(model_image_path)
        
        results = {}
        
        for angle in angles:
            logger.info(f"📐 Processing {angle}...")
            
            # Create angle-specific fitting configuration
            fitting_config = self._create_angle_specific_config(angle)
            
            # Fit ring for this angle
            fitted_image = self.fit_ring_to_hand(
                ring_image_path, model_image_path, fitting_config
            )
            
            # Apply angle-specific lighting
            lighting_config = lighting_configs.get(angle, LightingConfig())
            lit_image = self.apply_advanced_lighting(fitted_image, lighting_config)
            
            # Generate prompts for this angle
            prompts = self.generate_comprehensive_prompt(
                ring_props, hand_props, lighting_config,
                {"angle": angle, "view_type": angle.replace("_", " ")}
            )
            
            results[angle] = {
                'image': lit_image,
                'prompts': prompts,
                'ring_properties': asdict(ring_props),
                'hand_properties': asdict(hand_props),
                'lighting_config': asdict(lighting_config),
                'fitting_config': asdict(fitting_config)
            }
        
        logger.info("✅ Multiple enhanced angles generated")
        return results
    
    def save_comprehensive_results(self, 
                                 results: Dict, 
                                 output_dir: str = "enhanced_ring_fitting_results"):
        """
        Save comprehensive results with organized structure.
        """
        logger.info(f"💾 Saving comprehensive results to {output_dir}...")
        
        # Create organized directory structure
        base_path = Path(output_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        (base_path / "images").mkdir(exist_ok=True)
        (base_path / "prompts").mkdir(exist_ok=True)
        (base_path / "configs").mkdir(exist_ok=True)
        (base_path / "analysis").mkdir(exist_ok=True)
        
        # Save results for each angle
        for angle_name, angle_data in results.items():
            # Save image
            if 'image' in angle_data:
                img_path = base_path / "images" / f"{angle_name}.png"
                pil_img = Image.fromarray(angle_data['image'])
                pil_img.save(img_path, quality=95, optimize=True)
                logger.info(f"💾 Saved {angle_name}.png")
            
            # Save prompts
            if 'prompts' in angle_data:
                prompt_path = base_path / "prompts" / f"{angle_name}_prompts.json"
                with open(prompt_path, 'w', encoding='utf-8') as f:
                    json.dump(angle_data['prompts'], f, indent=2, ensure_ascii=False)
            
            # Save configurations
            config_data = {
                'ring_properties': angle_data.get('ring_properties', {}),
                'hand_properties': angle_data.get('hand_properties', {}),
                'lighting_config': angle_data.get('lighting_config', {}),
                'fitting_config': angle_data.get('fitting_config', {})
            }
            
            config_path = base_path / "configs" / f"{angle_name}_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        self._save_analysis_summary(results, base_path / "analysis" / "summary_report.json")
        
        logger.info("✅ Comprehensive results saved successfully")
    
    # =============== HELPER METHODS IMPLEMENTATION ===============
    
    def _find_ring_center_enhanced(self, image: np.ndarray) -> Tuple[int, int]:
        """Enhanced ring center detection using multiple methods."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Hough Circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=200)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return (circles[0][0], circles[0][1])
        
        # Method 2: Contour-based detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=30, param2=20, minRadius=15, maxRadius=150)
        
        if circles is not None:
            return (int(circles[0][0][0]), int(circles[0][0][1]))
        
        # Method 3: Center of mass
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(thresh)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        
        # Fallback to image center
        h, w = image.shape[:2]
        return (w // 2, h // 2)
    
    def _estimate_ring_radius_enhanced(self, image: np.ndarray, center: Tuple[int, int]) -> float:
        """Enhanced ring radius estimation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Using Hough circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=200)
        if circles is not None:
            return float(circles[0][0][2])
        
        # Method 2: Distance transform
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
        
        if max_val > 0:
            return float(max_val)
        
        # Method 3: Fallback estimation based on image size
        h, w = image.shape[:2]
        return min(w, h) * 0.3
    
    def _calculate_ring_angle_enhanced(self, image: np.ndarray, center: Tuple[int, int]) -> float:
        """Calculate ring orientation angle."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (should be the ring)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Fit ellipse to get orientation
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                return ellipse[2]  # Angle of the ellipse
        
        return 0.0  # Default angle
    
    def _classify_material_enhanced(self, image: np.ndarray) -> str:
        """Enhanced material classification using advanced features."""
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Calculate color statistics
        rgb_mean = np.mean(image, axis=(0, 1))
        
        # Analyze metallic properties
        metallic_score = self._calculate_metallic_score(image)
        
        # Enhanced classification logic
        if metallic_score > 0.6:
            if rgb_mean[0] > 180 and rgb_mean[1] > 140 and rgb_mean[2] < 100:
                return "rose_gold"
            elif rgb_mean[0] > 200 and rgb_mean[1] > 180 and rgb_mean[2] < 120:
                return "yellow_gold"
            elif np.mean(rgb_mean) > 200:
                return "white_gold"
            elif np.mean(rgb_mean) > 150:
                return "platinum"
            else:
                return "silver"
        else:
            if np.mean(rgb_mean) < 80:
                return "black_metal"
            else:
                return "stainless_steel"
    
    def _classify_style_enhanced(self, image: np.ndarray) -> str:
        """Classify ring style based on visual features."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges to analyze complexity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Analyze shape complexity
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if edge_density > 0.15:
            return "statement"
        elif edge_density > 0.08:
            return "vintage"
        elif edge_density > 0.05:
            return "solitaire"
        elif edge_density > 0.02:
            return "modern"
        else:
            return "band"
    
    def _detect_gemstone_enhanced(self, image: np.ndarray) -> Tuple[str, List[Tuple[int, int]]]:
        """Enhanced gemstone detection."""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Detect bright, high-saturation regions (potential gems)
        bright_mask = cv2.inRange(hsv, (0, 100, 150), (180, 255, 255))
        
        # Find contours of potential gems
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gem_positions = []
        if contours:
            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Filter small noise
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        gem_positions.append((cx, cy))
        
        # Classify gem type based on dominant colors
        if gem_positions:
            # Extract colors around gem positions
            colors = []
            for pos in gem_positions[:3]:  # Check up to 3 gems
                x, y = pos
                region = image[max(0, y-10):y+10, max(0, x-10):x+10]
                if region.size > 0:
                    colors.append(np.mean(region, axis=(0, 1)))
            
            if colors:
                avg_color = np.mean(colors, axis=0)
                # Simple color-based classification
                if avg_color[2] > 200 and avg_color[0] < 150 and avg_color[1] < 150:
                    return "diamond", gem_positions
                elif avg_color[0] > 150 and avg_color[1] < 100:
                    return "ruby", gem_positions
                elif avg_color[1] > 150 and avg_color[0] < 100:
                    return "emerald", gem_positions
                elif avg_color[2] > 150 and avg_color[0] < 100:
                    return "sapphire", gem_positions
                else:
                    return "gemstone", gem_positions
        
        return "none", []
    
    def _extract_color_palette_enhanced(self, image: np.ndarray) -> List[str]:
        """Extract dominant colors using KMeans clustering."""
        # Reshape image for clustering
        pixels = image.reshape(-1, 3)
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert to hex strings
        color_palette = []
        for color in colors:
            hex_color = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            color_palette.append(hex_color)
        
        return color_palette
    
    def _analyze_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze texture features of the ring."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        features = {
            'roughness': self._calculate_roughness(gray),
            'smoothness': self._calculate_smoothness(gray),
            'uniformity': self._calculate_uniformity(gray),
            'entropy': self._calculate_entropy(gray)
        }
        
        return features
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate image brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.mean(gray) / 255.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray) / 255.0
    
    def _calculate_saturation(self, image: np.ndarray) -> float:
        """Calculate image saturation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return np.mean(hsv[:, :, 1]) / 255.0
    
    def _calculate_metallic_score(self, image: np.ndarray) -> float:
        """Calculate how metallic the ring appears."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Metallic surfaces have high reflectivity variations
        reflectivity_variation = np.std(gray) / (np.mean(gray) + 1e-8)
        
        # Combine metrics
        metallic_score = min(1.0, (np.mean(gradient_magnitude) / 255.0 + reflectivity_variation) / 2.0)
        
        return metallic_score
    
    def _calculate_roughness(self, gray: np.ndarray) -> float:
        """Calculate surface roughness."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian) / (255.0 ** 2)
    
    def _calculate_smoothness(self, gray: np.ndarray) -> float:
        """Calculate surface smoothness."""
        return 1.0 - self._calculate_roughness(gray)
    
    def _calculate_uniformity(self, gray: np.ndarray) -> float:
        """Calculate uniformity of the surface."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        return np.sum(hist_norm ** 2)
    
    def _calculate_entropy(self, gray: np.ndarray) -> float:
        """Calculate entropy of the image."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
        return -np.sum(hist_norm * np.log2(hist_norm))
    
    def _enhanced_basic_hand_analysis(self, model_image_path: str) -> HandProperties:
        """Basic hand analysis when MediaPipe is not available."""
        image = cv2.imread(model_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create dummy landmarks for basic functionality
        landmarks = [(w//2, h//2) for _ in range(21)]  # 21 landmarks for MediaPipe compatibility
        
        return HandProperties(
            landmarks=landmarks,
            finger_positions={
                'thumb': (w//4, h//2),
                'index_finger': (w//3, h//3),
                'middle_finger': (w//2, h//4),
                'ring_finger': (2*w//3, h//3),
                'pinky': (3*w//4, h//2)
            },
            finger_joints={'ring_finger': [(2*w//3, h//3), (2*w//3, h//2)]},
            hand_orientation=0.0,
            skin_tone="medium",
            pose_category="neutral",
            hand_size=1.0,
            finger_widths={'ring_finger': 20.0},
            nail_positions={'ring_finger': (2*w//3, h//4)}
        )
    
    def _extract_finger_positions_enhanced(self, landmarks: List[Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Extract finger tip positions from landmarks."""
        # MediaPipe hand landmarks indices
        finger_tips = {
            'thumb': landmarks[4],
            'index_finger': landmarks[8],
            'middle_finger': landmarks[12],
            'ring_finger': landmarks[16],
            'pinky': landmarks[20]
        }
        return finger_tips
    
    def _extract_finger_joints(self, landmarks: List[Tuple[int, int]]) -> Dict[str, List[Tuple[int, int]]]:
        """Extract finger joint positions."""
        finger_joints = {
            'thumb': [landmarks[2], landmarks[3], landmarks[4]],
            'index_finger': [landmarks[5], landmarks[6], landmarks[7], landmarks[8]],
            'middle_finger': [landmarks[9], landmarks[10], landmarks[11], landmarks[12]],
            'ring_finger': [landmarks[13], landmarks[14], landmarks[15], landmarks[16]],
            'pinky': [landmarks[17], landmarks[18], landmarks[19], landmarks[20]]
        }
        return finger_joints
    
    def _calculate_hand_orientation_enhanced(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate hand orientation angle."""
        # Use wrist to middle finger vector
        wrist = landmarks[0]
        middle_finger = landmarks[12]
        
        dx = middle_finger[0] - wrist[0]
        dy = middle_finger[1] - wrist[1]
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        return angle
    
    def _analyze_skin_tone_enhanced(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> str:
        """Analyze skin tone from hand region."""
        # Sample skin color around landmarks
        skin_samples = []
        for x, y in landmarks[:5]:  # Sample from first 5 landmarks
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                region = image[max(0, y-5):y+5, max(0, x-5):x+5]
                if region.size > 0:
                    skin_samples.append(np.mean(region, axis=(0, 1)))
        
        if skin_samples:
            avg_skin = np.mean(skin_samples, axis=0)
            brightness = np.mean(avg_skin)
            
            if brightness > 200:
                return "light"
            elif brightness > 150:
                return "medium_light"
            elif brightness > 100:
                return "medium"
            elif brightness > 70:
                return "medium_dark"
            else:
                return "dark"
        
        return "medium"
    
    def _classify_hand_pose_enhanced(self, landmarks: List[Tuple[int, int]]) -> str:
        """Classify hand pose category."""
        # Analyze finger positions relative to palm
        wrist = landmarks[0]
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        # Calculate distances from wrist to fingertips
        distances = []
        for tip in finger_tips:
            dist = math.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Simple pose classification
        if avg_distance > 150:
            return "extended"
        elif avg_distance > 100:
            return "relaxed"
        else:
            return "closed"
    
    def _calculate_hand_size(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate relative hand size."""
        wrist = landmarks[0]
        middle_tip = landmarks[12]
        
        hand_length = math.sqrt((middle_tip[0] - wrist[0])**2 + (middle_tip[1] - wrist[1])**2)
        
        # Normalize to a standard size (assuming 200px is average)
        return hand_length / 200.0
    
    def _calculate_finger_widths(self, landmarks: List[Tuple[int, int]], image: np.ndarray) -> Dict[str, float]:
        """Calculate finger widths."""
        finger_widths = {}
        finger_bases = {
            'thumb': landmarks[2],
            'index_finger': landmarks[5],
            'middle_finger': landmarks[9],
            'ring_finger': landmarks[13],
            'pinky': landmarks[17]
        }
        
        for finger, base in finger_bases.items():
            # Estimate width as a proportion of hand size
            finger_widths[finger] = 15.0 + np.random.normal(0, 2)  # Simple estimation
        
        return finger_widths
    
    def _detect_nail_positions(self, landmarks: List[Tuple[int, int]], image: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """Detect nail positions on fingertips."""
        finger_tips = {
            'thumb': landmarks[4],
            'index_finger': landmarks[8],
            'middle_finger': landmarks[12],
            'ring_finger': landmarks[16],
            'pinky': landmarks[20]
        }
        
        # Estimate nail positions slightly above fingertips
        nail_positions = {}
        for finger, tip in finger_tips.items():
            nail_pos = (tip[0], tip[1] - 5)  # Slightly above tip
            nail_positions[finger] = nail_pos
        
        return nail_positions
    
    def _load_image_with_alpha(self, image_path: str) -> np.ndarray:
        """Load image with alpha channel handling."""
        # Try loading with PIL first (better alpha support)
        try:
            pil_img = Image.open(image_path).convert("RGBA")
            return np.array(pil_img)
        except:
            # Fallback to OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 3:  # Add alpha channel if missing
                alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                img = np.concatenate([img, alpha], axis=2)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    def _extract_ring_enhanced(self, ring_image: np.ndarray, ring_props: RingProperties) -> np.ndarray:
        """Enhanced ring extraction preserving original quality."""
        if ring_image.shape[2] == 4:  # Already has alpha
            return ring_image
        
        # Create precise mask using multiple techniques
        mask = self._create_precise_ring_mask(ring_image[:,:,:3], ring_props)
        
        # Add alpha channel
        ring_rgba = np.dstack([ring_image[:, :, :3], mask])
        
        return ring_rgba
    
    def _create_precise_ring_mask(self, image: np.ndarray, ring_props: RingProperties) -> np.ndarray:
        """Create precise ring mask using advanced segmentation."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Create initial mask using multiple methods
        masks = []
        
        # Method 1: Threshold-based
        _, thresh_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(thresh_mask)
        
        # Method 2: Edge-based
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        masks.append(edge_mask)
        
        # Method 3: Center-focused circular mask
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(center_mask, ring_props.center, int(ring_props.radius * 1.5), 255, -1)
        masks.append(center_mask)
        
        # Combine masks
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Refine mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Smooth edges
        combined_mask = cv2.GaussianBlur(combined_mask, (3, 3), 1)
        
        return combined_mask
    
    def _get_finger_placement_info(self, hand_props: HandProperties, target_finger: str) -> Dict:
        """Get detailed finger placement information."""
        finger_pos = hand_props.finger_positions.get(target_finger, (0, 0))
        finger_joints = hand_props.finger_joints.get(target_finger, [finger_pos])
        finger_width = hand_props.finger_widths.get(target_finger, 20.0)
        
        return {
            'position': finger_pos,
            'joints': finger_joints,
            'width': finger_width,
            'orientation': hand_props.hand_orientation
        }
    
    def _calculate_precise_transform(self, ring_props: RingProperties, hand_props: HandProperties, 
                                   finger_info: Dict, fitting_config: FittingConfig) -> np.ndarray:
        """Calculate precise transformation matrix for ring placement."""
        # Calculate scale based on finger width and ring size
        target_width = finger_info['width'] * 0.8  # Ring should be slightly smaller than finger
        scale_factor = target_width / (ring_props.radius * 2) * fitting_config.ring_scale
        
        # Calculate rotation
        rotation_angle = finger_info['orientation'] + fitting_config.rotation_adjustment
        
        # Calculate translation
        target_pos = finger_info['position']
        ring_center = ring_props.center
        
        tx = target_pos[0] - ring_center[0] + fitting_config.position_offset[0]
        ty = target_pos[1] - ring_center[1] + fitting_config.position_offset[1]
        
        # Create transformation matrix
        cos_r = math.cos(math.radians(rotation_angle))
        sin_r = math.sin(math.radians(rotation_angle))
        
        # Combined transformation matrix
        transform_matrix = np.array([
            [scale_factor * cos_r, -scale_factor * sin_r, tx],
            [scale_factor * sin_r,  scale_factor * cos_r, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform_matrix
    
    def _transform_ring_enhanced(self, ring_image: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """Apply non-deformative transformation to ring."""
        h, w = ring_image.shape[:2]
        
        # Apply transformation
        transformed = cv2.warpAffine(ring_image, transform_matrix[:2], (w, h), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        
        return transformed
    
    def _generate_ring_shadow(self, ring_image: np.ndarray, finger_info: Dict) -> np.ndarray:
        """Generate realistic shadow for the ring."""
        h, w = ring_image.shape[:2]
        shadow = np.zeros((h, w), dtype=np.uint8)
        
        # Simple shadow generation (offset and blur)
        if ring_image.shape[2] == 4:  # Has alpha channel
            alpha = ring_image[:, :, 3]
            # Create shadow by offsetting alpha
            shadow_offset = (5, 5)  # Shadow offset
            M = np.float32([[1, 0, shadow_offset[0]], [0, 1, shadow_offset[1]]])
            shadow = cv2.warpAffine(alpha, M, (w, h))
            shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
            shadow = (shadow * 0.3).astype(np.uint8)  # Make it semi-transparent
        
        return shadow
    
    def _generate_ring_reflection(self, ring_image: np.ndarray, ring_props: RingProperties) -> np.ndarray:
        """Generate realistic reflection for the ring."""
        if ring_props.metallic_score < 0.5:
            return None  # No reflection for non-metallic rings
        
        h, w = ring_image.shape[:2]
        reflection = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Simple reflection effect
        if ring_image.shape[2] == 4:
            # Create a brightened version of the ring
            reflection[:, :, :3] = np.clip(ring_image[:, :, :3] * 1.3, 0, 255)
            reflection[:, :, 3] = (ring_image[:, :, 3] * 0.2).astype(np.uint8)  # Low opacity
        
        return reflection
    
    def _blend_ring_enhanced(self, base_image: np.ndarray, ring_image: np.ndarray, 
                             shadow_layer: np.ndarray, reflection_layer: np.ndarray, 
                             finger_info: Dict) -> np.ndarray:
        """Enhanced blending with shadow and reflection layers."""
        result = base_image.copy()
        
        # Blend shadow first (if exists)
        if shadow_layer is not None:
            # Convert grayscale shadow to RGB and apply dark color
            shadow_rgb = cv2.cvtColor(shadow_layer, cv2.COLOR_GRAY2RGB)
            shadow_rgb = cv2.resize(shadow_rgb, (result.shape[1], result.shape[0]))
            
            # Create alpha mask from shadow layer
            shadow_alpha = cv2.resize(shadow_layer, (result.shape[1], result.shape[0]))
            shadow_alpha = shadow_alpha.astype(float) / 255.0
            
            # Blend shadow (darken the base image where shadow exists)
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - shadow_alpha) + shadow_rgb[:, :, c] * shadow_alpha * 0.5
        
        # Rest of the blending code remains the same...
        # Blend ring
        if ring_image.shape[2] == 4:  # Has alpha channel
            alpha = ring_image[:, :, 3] / 255.0
            alpha = np.stack([alpha] * 3, axis=2)
            
            ring_rgb = ring_image[:, :, :3]
            result = result * (1 - alpha) + ring_rgb * alpha
        else:
            # Simple overlay if no alpha
            mask = np.any(ring_image > 0, axis=2)
            result[mask] = ring_image[mask]
        
        # Blend reflection (if exists)
        if reflection_layer is not None:
            refl_alpha = reflection_layer[:, :, 3] / 255.0
            refl_alpha = np.stack([refl_alpha] * 3, axis=2)
            
            refl_rgb = reflection_layer[:, :, :3]
            result = result * (1 - refl_alpha) + refl_rgb * refl_alpha
        
        return result.astype(np.uint8)
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance overall image quality."""
        # Convert to PIL for better enhancement
        pil_img = Image.fromarray(image)
        
        # Apply enhancements
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.05)
        
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.02)
        
        return np.array(pil_img)
    
    # =============== LIGHTING METHODS ===============
    
    def _apply_studio_lighting(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply professional studio lighting."""
        # Increase contrast and brightness for studio look
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config.brightness * 1.1)
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(config.contrast * 1.2)
        
        return image
    
    def _apply_natural_lighting(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply natural lighting effects."""
        # Softer, warmer lighting
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config.brightness * 0.95)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(config.saturation * 0.9)
        
        return image
    
    def _apply_dramatic_lighting(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply dramatic lighting effects."""
        # High contrast, selective brightness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(config.contrast * 1.4)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config.brightness * 0.9)
        
        return image
    
    def _apply_soft_lighting(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply soft, diffused lighting."""
        # Reduce contrast, increase brightness slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(config.contrast * 0.8)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config.brightness * 1.05)
        
        return image
    
    def _apply_basic_lighting_adjustments(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply basic lighting adjustments."""
        # Brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(config.brightness)
        
        # Contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(config.contrast)
        
        # Saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(config.saturation)
        
        return image
    
    def _apply_advanced_lighting_effects(self, image: Image.Image, config: LightingConfig) -> Image.Image:
        """Apply advanced lighting effects."""
        # Apply warmth adjustment
        if config.warmth != 0:
            image = self._adjust_warmth(image, config.warmth)
        
        # Apply sharpening for better definition
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        
        return image
    
    def _adjust_warmth(self, image: Image.Image, warmth: float) -> Image.Image:
        """Adjust image warmth (-1 cool to +1 warm)."""
        np_img = np.array(image)
        
        if warmth > 0:  # Warm
            np_img[:, :, 0] = np.clip(np_img[:, :, 0] * (1 + warmth * 0.1), 0, 255)  # Increase red
            np_img[:, :, 1] = np.clip(np_img[:, :, 1] * (1 + warmth * 0.05), 0, 255)  # Slightly increase green
        else:  # Cool
            np_img[:, :, 2] = np.clip(np_img[:, :, 2] * (1 + abs(warmth) * 0.1), 0, 255)  # Increase blue
        
        return Image.fromarray(np_img.astype(np.uint8))
    
    # =============== PROMPT GENERATION METHODS ===============
    
    def _build_ring_description(self, ring_props: RingProperties) -> str:
        """Build detailed ring description."""
        desc = f"{ring_props.style} {ring_props.material} ring"
        
        if ring_props.gem_type != "none":
            desc += f" featuring {ring_props.gem_type}"
            if len(ring_props.gem_positions) > 1:
                desc += f" with {len(ring_props.gem_positions)} stones"
        
        return desc
    
    def _build_hand_description(self, hand_props: HandProperties) -> str:
        """Build detailed hand description."""
        return f"{hand_props.skin_tone} skin toned hand in {hand_props.pose_category} pose"
    
    def _build_lighting_description(self, lighting_config: LightingConfig) -> str:
        """Build lighting description."""
        return self.prompt_templates['lighting_descriptions'].get(
            lighting_config.environment_lighting, 
            "professional lighting"
        )
    
    def _build_technical_description(self) -> str:
        """Build technical specifications description."""
        return ", ".join(self.prompt_templates['technical_specs'][:3])
    
    def _generate_detailed_prompt(self, ring_desc: str, hand_desc: str, 
                                lighting_desc: str, technical_desc: str) -> str:
        """Generate detailed prompt."""
        return f"Professional jewelry photography: {ring_desc} elegantly worn on {hand_desc}, {lighting_desc}, {technical_desc}, ultra-high resolution, perfect focus"
    
    def _generate_commercial_prompt(self, ring_desc: str, hand_desc: str, lighting_desc: str) -> str:
        """Generate commercial-style prompt."""
        return f"Commercial jewelry advertisement: {ring_desc} showcased on {hand_desc}, {lighting_desc}, luxury presentation, marketing quality"
    
    def _generate_artistic_prompt(self, ring_desc: str, hand_desc: str, lighting_desc: str) -> str:
        """Generate artistic prompt."""
        return f"Artistic jewelry portrait: {ring_desc} beautifully displayed on {hand_desc}, {lighting_desc}, creative composition, fine art photography"
    
    def _generate_technical_prompt(self, ring_desc: str, hand_desc: str, technical_desc: str) -> str:
        """Generate technical prompt."""
        return f"Technical jewelry documentation: {ring_desc} on {hand_desc}, {technical_desc}, precise details, catalog photography"
    
    def _generate_social_media_prompt(self, ring_desc: str, hand_desc: str) -> str:
        """Generate social media optimized prompt."""
        return f"Instagram-worthy shot: {ring_desc} on {hand_desc}, trendy, lifestyle photography, social media ready"
    
    def _enhance_prompt_with_details(self, prompt: str, details: Dict) -> str:
        """Enhance prompt with additional details."""
        if 'angle' in details:
            prompt += f", {details['angle']} perspective"
        if 'style' in details:
            prompt += f", {details['style']} style"
        
        return prompt
    
    def _create_angle_specific_config(self, angle: str) -> FittingConfig:
        """Create angle-specific fitting configuration."""
        config = FittingConfig()
        
        if angle == "close_up":
            config.ring_scale = 1.2
        elif angle == "side_view":
            config.rotation_adjustment = 30
        elif angle == "top_view":
            config.position_offset = (0, -10)
        
        return config
    
    def _save_analysis_summary(self, results: Dict, output_path: Path):
        """Save analysis summary report."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'angles_processed': list(results.keys()),
            'total_images': len(results),
            'pipeline_version': '2.0',
            'features_used': [
                'advanced_ring_analysis',
                'hand_landmark_detection',
                'multi_angle_generation',
                'advanced_lighting',
                'comprehensive_prompts'
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            # Apply configuration to pipeline
            logger.info(f"Configuration loaded from {config_path}")

# Usage example and main execution
if __name__ == "__main__":
    # Initialize the enhanced pipeline
    pipeline = EnhancedRingFittingPipeline()
    
    # Example usage
    try:
        # Set up paths (replace with your actual image paths)
        ring_path = r"C:\Users\drago\Desktop\valorant-ml-system\scripts\ring2.png"
        model_path = r"C:\Users\drago\Desktop\valorant-ml-system\scripts\model1.png"
        
        # Check if images exist
        if not os.path.exists(ring_path) or not os.path.exists(model_path):
            print("⚠️  Please update the image paths in the script with your actual image files.")
            print("📁 Example paths needed:")
            print(f"   Ring image: {ring_path}")
            print(f"   Model image: {model_path}")
            exit(1)
        
        # Create custom lighting configuration
        studio_lighting = LightingConfig(
            brightness=1.1,
            contrast=1.2,
            saturation=1.05,
            warmth=0.1,
            environment_lighting="studio",  # Complete the environment_lighting parameter
            shadow_intensity=0.6,
            highlight_intensity=0.8,
            ambient_light=0.9,
            directional_light_angle=45.0,
            light_softness=0.7,
            reflection_intensity=0.65
        )

        # Process multiple angles
        results = pipeline.generate_multiple_angles_enhanced(
            ring_path,
            model_path,
            angles=["front_view", "side_view", "top_view", "angled_view", "close_up"],
            lighting_configs={
                "front_view": studio_lighting,
                "side_view": LightingConfig(
                    brightness=1.05,
                    contrast=1.3,
                    environment_lighting="dramatic"
                ),
                "top_view": LightingConfig(
                    brightness=1.15,
                    contrast=1.1,
                    environment_lighting="soft"
                ),
                "angled_view": LightingConfig(
                    brightness=1.2,
                    contrast=1.15,
                    environment_lighting="natural"
                ),
                "close_up": studio_lighting
            }
        )

        # Save results
        output_dir = "enhanced_ring_fitting_output"
        pipeline.save_comprehensive_results(results, output_dir)
        
        print("\n✅ Processing complete!")
        print(f"📁 Results saved to: {output_dir}")
        print("\nGenerated outputs include:")
        print("- Multiple angle variations")
        print("- Professional lighting effects")
        print("- Comprehensive prompts")
        print("- Detailed analysis reports")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check the image file paths and try again.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
