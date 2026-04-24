# Requirements Document

## Introduction

This document specifies requirements for improving the PDF image extraction system in a medical histology application. The current implementation uses basic PyMuPDF extraction with minimal preprocessing. The improved system will incorporate image quality enhancements, intelligent magnification, enhanced preprocessing, and robust fallback mechanisms based on the reference implementation from the mueva_test repository. Image quality is critical in this medical context as extracted images are used for diagnostic visual analysis and embedding-based retrieval.

## Glossary

- **Image_Extractor**: The ExtractorImagenesPDF class responsible for extracting images from PDF files
- **Source_PDF**: A PDF file containing medical histology images and text
- **Extracted_Image**: A PNG image file extracted from a Source_PDF page
- **Caption**: Descriptive text located spatially below an image on the same page
- **Page_Text**: The complete text content of a PDF page used for text-based searches
- **Minimum_Size_Threshold**: The minimum width and height (in pixels) for an image to be considered valid
- **Target_Magnification_Size**: The target dimension (868 pixels) for upscaling small but valid images
- **Preprocessing_Pipeline**: The sequence of image enhancement operations applied to extracted images
- **Fallback_Renderer**: The pdf2image-based page rendering system used when PyMuPDF extraction fails
- **LANCZOS_Resampling**: A high-quality image resampling algorithm used for magnification

## Requirements

### Requirement 1: Image Size Filtering

**User Story:** As a medical researcher, I want the system to filter out small icons and graphics, so that only meaningful histological images are extracted.

#### Acceptance Criteria

1. WHEN an image is extracted from a Source_PDF, THE Image_Extractor SHALL reject images with width less than 150 pixels
2. WHEN an image is extracted from a Source_PDF, THE Image_Extractor SHALL reject images with height less than 150 pixels
3. WHEN multiple valid images exist on a page, THE Image_Extractor SHALL select the image with the largest area (width × height)
4. FOR ALL extracted images, the width SHALL be greater than or equal to 150 pixels AND the height SHALL be greater than or equal to 150 pixels

### Requirement 2: Image Magnification for Embedding Quality

**User Story:** As a system administrator, I want small but valid images to be magnified to a standard size, so that embedding quality is consistent across all images.

#### Acceptance Criteria

1. WHEN an Extracted_Image has width less than 868 pixels, THE Image_Extractor SHALL magnify the image to 868 pixels width while preserving aspect ratio
2. WHEN an Extracted_Image has height less than 868 pixels AND width is greater than or equal to 868 pixels, THE Image_Extractor SHALL magnify the image to 868 pixels height while preserving aspect ratio
3. WHEN magnifying an image, THE Image_Extractor SHALL use LANCZOS_Resampling algorithm
4. WHEN an Extracted_Image has both width and height greater than or equal to 868 pixels, THE Image_Extractor SHALL NOT magnify the image
5. FOR ALL magnified images, the aspect ratio SHALL remain unchanged (within 1% tolerance)

### Requirement 3: Image Preprocessing Enhancement

**User Story:** As a medical researcher, I want extracted images to have enhanced contrast and brightness, so that histological features are more visible for analysis.

#### Acceptance Criteria

1. WHERE contrast enhancement is enabled, THE Preprocessing_Pipeline SHALL apply contrast enhancement with factor 1.2 to all Extracted_Images
2. WHERE brightness enhancement is enabled, THE Preprocessing_Pipeline SHALL apply brightness enhancement with factor 1.1 to all Extracted_Images
3. THE Image_Extractor SHALL apply contrast enhancement before brightness enhancement
4. WHEN preprocessing fails for an image, THE Image_Extractor SHALL save the unprocessed image and log a warning
5. FOR ALL preprocessed images, the pixel value range SHALL remain within valid bounds (0-255 for 8-bit images)

### Requirement 4: Configuration-Driven Enhancement

**User Story:** As a system administrator, I want to configure image enhancement settings, so that I can tune image quality based on specific use cases.

#### Acceptance Criteria

1. THE Image_Extractor SHALL read contrast enhancement configuration from a Config.ENHANCE_CONTRAST setting
2. THE Image_Extractor SHALL read brightness enhancement configuration from a Config.ENHANCE_BRIGHTNESS setting
3. WHEN Config.ENHANCE_CONTRAST is False, THE Preprocessing_Pipeline SHALL skip contrast enhancement
4. WHEN Config.ENHANCE_BRIGHTNESS is False, THE Preprocessing_Pipeline SHALL skip brightness enhancement
5. THE Image_Extractor SHALL provide default values (True for both settings) when configuration is not specified

### Requirement 5: Robust Fallback Rendering

**User Story:** As a medical researcher, I want the system to render the entire page as an image when extraction fails, so that no pages are skipped due to technical issues.

#### Acceptance Criteria

1. WHEN PyMuPDF extraction finds no valid images on a page, THE Image_Extractor SHALL invoke the Fallback_Renderer
2. WHEN the Fallback_Renderer is invoked, THE Image_Extractor SHALL use pdf2image with DPI configuration
3. WHEN the Fallback_Renderer produces an image, THE Image_Extractor SHALL apply the same Preprocessing_Pipeline as extracted images
4. WHEN the Fallback_Renderer produces an image, THE Image_Extractor SHALL apply the same magnification rules as extracted images
5. IF the Fallback_Renderer fails, THEN THE Image_Extractor SHALL log an error and continue processing remaining pages

### Requirement 6: Caption Extraction Preservation

**User Story:** As a medical researcher, I want descriptive text below images to be extracted as captions, so that I can understand what tissue or structure the image represents.

#### Acceptance Criteria

1. THE Image_Extractor SHALL preserve the existing caption extraction mechanism that extracts text spatially below images
2. WHEN extracting a caption, THE Image_Extractor SHALL use the image bounding box to determine spatial position
3. WHEN extracting a caption, THE Image_Extractor SHALL extract all text from below the image to the end of the page
4. WHEN no text exists below an image, THE Image_Extractor SHALL use the first 500 characters of Page_Text as fallback
5. THE Image_Extractor SHALL NOT extract text from above the image as part of the caption

### Requirement 7: Page Text Preservation

**User Story:** As a medical researcher, I want the complete text of each page to be preserved, so that I can perform text-based searches across the document.

#### Acceptance Criteria

1. THE Image_Extractor SHALL preserve the existing Page_Text extraction mechanism
2. FOR ALL pages with extracted images, THE Image_Extractor SHALL extract and store the complete Page_Text
3. THE Image_Extractor SHALL extract Page_Text from only the current page, not adjacent pages
4. WHEN Page_Text extraction fails, THE Image_Extractor SHALL store an empty string and continue processing

### Requirement 8: Error Handling and Logging

**User Story:** As a system administrator, I want detailed error logging during image extraction, so that I can diagnose and fix issues quickly.

#### Acceptance Criteria

1. WHEN an error occurs during image extraction, THE Image_Extractor SHALL log the error with page number and error details
2. WHEN an error occurs during preprocessing, THE Image_Extractor SHALL log a warning and save the unprocessed image
3. WHEN the Fallback_Renderer is invoked, THE Image_Extractor SHALL log the reason for fallback
4. WHEN an image is magnified, THE Image_Extractor SHALL log the original and target dimensions
5. IF a page fails completely, THEN THE Image_Extractor SHALL continue processing remaining pages without terminating

### Requirement 9: Output Format Consistency

**User Story:** As a developer, I want all extracted images to maintain the same output format and metadata structure, so that downstream processing is consistent.

#### Acceptance Criteria

1. THE Image_Extractor SHALL save all Extracted_Images in PNG format
2. THE Image_Extractor SHALL preserve the existing output metadata structure containing path, fuente_pdf, pagina, indice, ocr_text, texto_pagina, and caption fields
3. THE Image_Extractor SHALL preserve the existing file naming convention: {pdf_name}_pag{page_number}.png
4. THE Image_Extractor SHALL save all Extracted_Images to the configured output directory
5. FOR ALL extracted images, the metadata structure SHALL be identical regardless of extraction method (PyMuPDF or Fallback_Renderer)

### Requirement 10: Backward Compatibility

**User Story:** As a developer, I want the improved extraction system to maintain the same interface, so that existing code continues to work without modification.

#### Acceptance Criteria

1. THE Image_Extractor SHALL maintain the existing class name ExtractorImagenesPDF
2. THE Image_Extractor SHALL maintain the existing method signature for extraer_de_pdf(pdf_path: str) -> List[Dict[str, str]]
3. THE Image_Extractor SHALL maintain the existing constructor signature __init__(directorio_salida: str)
4. THE Image_Extractor SHALL maintain the existing static method extraer_caption_imagen with its current signature
5. THE Image_Extractor SHALL return the same metadata dictionary structure as the current implementation
