#!/usr/bin/env python3
"""
Test script for improved image extraction
"""
import os
import sys
import importlib.util

# Load the module from file
spec = importlib.util.spec_from_file_location("ne4j_histo", "ne4j-histo.py")
ne4j_histo = importlib.util.module_from_spec(spec)
sys.modules["ne4j_histo"] = ne4j_histo
spec.loader.exec_module(ne4j_histo)

# Import after loading
import pymupdf as fitz
from PIL import Image
from ne4j_histo import ExtractorImagenesPDF, ImageExtractionConfig

def test_extraction():
    """Test the improved image extraction"""
    print("="*80)
    print("🧪 Testing Improved Image Extraction")
    print("="*80)
    
    # Test 1: Configuration
    print("\n1️⃣ Testing Configuration...")
    config = ImageExtractionConfig()
    assert config.MIN_WIDTH == 150, f"Expected MIN_WIDTH=150, got {config.MIN_WIDTH}"
    assert config.MIN_HEIGHT == 150, f"Expected MIN_HEIGHT=150, got {config.MIN_HEIGHT}"
    assert config.TARGET_MAGNIFICATION_SIZE == 868, f"Expected TARGET_SIZE=868, got {config.TARGET_MAGNIFICATION_SIZE}"
    assert config.ENHANCE_CONTRAST == True, "ENHANCE_CONTRAST should be True"
    assert config.ENHANCE_BRIGHTNESS == True, "ENHANCE_BRIGHTNESS should be True"
    assert config.CONTRAST_FACTOR == 1.2, f"Expected CONTRAST_FACTOR=1.2, got {config.CONTRAST_FACTOR}"
    assert config.BRIGHTNESS_FACTOR == 1.1, f"Expected BRIGHTNESS_FACTOR=1.1, got {config.BRIGHTNESS_FACTOR}"
    print("   ✅ Configuration correct")
    
    # Test 2: Extractor initialization
    print("\n2️⃣ Testing Extractor Initialization...")
    test_dir = "./test_images_output"
    os.makedirs(test_dir, exist_ok=True)
    extractor = ExtractorImagenesPDF(test_dir, config)
    assert extractor.config.MIN_WIDTH == 150
    assert hasattr(extractor, '_apply_preprocessing'), "Missing _apply_preprocessing method"
    assert hasattr(extractor, '_apply_magnification'), "Missing _apply_magnification method"
    assert hasattr(extractor, '_fallback_render_page'), "Missing _fallback_render_page method"
    print("   ✅ Extractor initialized with all methods")
    
    # Test 3: Preprocessing
    print("\n3️⃣ Testing Preprocessing...")
    test_img = Image.new('RGB', (200, 200), color='red')
    processed = extractor._apply_preprocessing(test_img)
    assert processed.size == (200, 200), "Preprocessing should preserve size"
    print("   ✅ Preprocessing works")
    
    # Test 4: Magnification
    print("\n4️⃣ Testing Magnification...")
    # Small image should be magnified
    small_img = Image.new('RGB', (400, 300), color='blue')
    magnified = extractor._apply_magnification(small_img)
    assert magnified.width == 868, f"Expected width=868, got {magnified.width}"
    expected_height = int(300 * (868 / 400))
    assert abs(magnified.height - expected_height) <= 1, f"Expected height≈{expected_height}, got {magnified.height}"
    print(f"   ✅ Magnification works: {small_img.size} → {magnified.size}")
    
    # Large image should not be magnified
    large_img = Image.new('RGB', (1000, 1000), color='green')
    not_magnified = extractor._apply_magnification(large_img)
    assert not_magnified.size == (1000, 1000), "Large images should not be magnified"
    print(f"   ✅ Large images preserved: {large_img.size} → {not_magnified.size}")
    
    # Test 5: Real PDF extraction
    print("\n5️⃣ Testing Real PDF Extraction...")
    pdf_path = "pdf/arch2.pdf"
    if os.path.exists(pdf_path):
        print(f"   📄 Extracting from {pdf_path}...")
        results = extractor.extraer_de_pdf(pdf_path)
        print(f"   ✅ Extracted {len(results)} images")
        
        if results:
            first_img = results[0]
            print(f"\n   📊 First image metadata:")
            print(f"      - Path: {first_img['path']}")
            print(f"      - Page: {first_img['pagina']}")
            print(f"      - Caption length: {len(first_img['caption'])} chars")
            print(f"      - Page text length: {len(first_img['texto_pagina'])} chars")
            
            # Verify image was saved and has correct dimensions
            if os.path.exists(first_img['path']):
                img = Image.open(first_img['path'])
                print(f"      - Image dimensions: {img.size}")
                # Should be at least 868px in one dimension (after magnification)
                assert max(img.size) >= 868, f"Image should be magnified to at least 868px, got {img.size}"
                print(f"   ✅ Image correctly magnified and saved")
    else:
        print(f"   ⚠️ Test PDF not found: {pdf_path}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)

if __name__ == "__main__":
    test_extraction()
