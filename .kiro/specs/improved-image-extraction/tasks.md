# Implementation Plan: Improved Image Extraction

## Overview

This implementation plan converts the design for improved PDF image extraction into actionable coding tasks. The enhancement adds intelligent image quality improvements, magnification for consistent embedding quality, configurable preprocessing pipelines, and robust fallback mechanisms to the existing `ExtractorImagenesPDF` class in `ne4j-histo.py`.

**Key Implementation Strategy:**
- Maintain backward compatibility (same class interface)
- Add configuration class for all constants
- Implement preprocessing pipeline (contrast + brightness)
- Implement magnification engine (LANCZOS resampling)
- Enhance fallback renderer with same preprocessing
- Add comprehensive property-based and unit tests

**Target File:** `ne4j-histo.py` (ExtractorImagenesPDF class, lines 1150-1300)

## Tasks

- [x] 1. Create configuration infrastructure
  - Create `ImageExtractionConfig` class with all constants (MIN_WIDTH=150, MIN_HEIGHT=150, TARGET_MAGNIFICATION_SIZE=868, ENHANCE_CONTRAST=True, ENHANCE_BRIGHTNESS=True, CONTRAST_FACTOR=1.2, BRIGHTNESS_FACTOR=1.1, FALLBACK_DPI=150, RESAMPLING_ALGORITHM=Image.Resampling.LANCZOS)
  - Update `ExtractorImagenesPDF.__init__` to accept optional config parameter
  - Update MIN_WIDTH and MIN_HEIGHT from 200 to 150 in the class
  - _Requirements: 1.1, 1.2, 4.1, 4.2, 4.5_

- [ ]* 1.1 Write property test for configuration defaults
  - **Property: Configuration provides correct default values**
  - **Validates: Requirements 4.5**

- [ ] 2. Implement preprocessing pipeline
  - [x] 2.1 Create `_apply_preprocessing` method in ExtractorImagenesPDF
    - Add PIL ImageEnhance imports
    - Implement contrast enhancement with factor 1.2 (applied first)
    - Implement brightness enhancement with factor 1.1 (applied second)
    - Add configuration checks for ENHANCE_CONTRAST and ENHANCE_BRIGHTNESS flags
    - Add try-except error handling with fallback to original image
    - Add warning logging for preprocessing failures
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.3, 4.4, 8.2_

  - [ ]* 2.2 Write property test for pixel value bounds invariant
    - **Property 6: Pixel Value Bounds Invariant**
    - **Validates: Requirements 3.5**

  - [ ]* 2.3 Write unit tests for preprocessing pipeline
    - Test contrast enhancement with factor 1.2
    - Test brightness enhancement with factor 1.1
    - Test correct ordering (contrast before brightness)
    - Test configuration flags (ENHANCE_CONTRAST, ENHANCE_BRIGHTNESS)
    - Test error handling and fallback to original image
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.3, 4.4_

- [ ] 3. Implement magnification engine
  - [x] 3.1 Create `_apply_magnification` method in ExtractorImagenesPDF
    - Add dimension checking logic (check if w >= 868 and h >= 868)
    - Implement width-based magnification (if w < 868: scale to width=868)
    - Implement height-based magnification (if h < 868 and w >= 868: scale to height=868)
    - Use LANCZOS resampling algorithm from config
    - Preserve aspect ratio in all magnification operations
    - Add logging for magnification operations (original and target dimensions)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 8.4_

  - [ ]* 3.2 Write property test for width-based magnification
    - **Property 3: Width-Based Magnification**
    - **Validates: Requirements 2.1, 2.5**

  - [ ]* 3.3 Write property test for height-based magnification
    - **Property 4: Height-Based Magnification**
    - **Validates: Requirements 2.2, 2.5**

  - [ ]* 3.4 Write property test for no magnification on large images
    - **Property 5: No Magnification for Large Images**
    - **Validates: Requirements 2.4**

  - [ ]* 3.5 Write unit tests for magnification engine
    - Test LANCZOS resampling algorithm is used
    - Test aspect ratio preservation (within 1% tolerance)
    - Test logging of magnification operations
    - _Requirements: 2.3, 2.5, 8.4_

- [x] 4. Checkpoint - Verify preprocessing and magnification work independently
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Enhance fallback renderer
  - [x] 5.1 Refactor existing fallback code into `_fallback_render_page` method
    - Extract fallback rendering logic from `extraer_de_pdf` into separate method
    - Accept pdf_path and page_num as parameters
    - Return PIL Image object or None on failure
    - Use pdf2image with DPI from config (FALLBACK_DPI=150)
    - Add logging for fallback invocation with reason
    - Add error handling and logging for fallback failures
    - _Requirements: 5.1, 5.2, 5.5, 8.3, 8.5_

  - [x] 5.2 Apply preprocessing pipeline to fallback images
    - Call `_apply_preprocessing` on fallback-rendered images
    - _Requirements: 5.3_

  - [x] 5.3 Apply magnification logic to fallback images
    - Call `_apply_magnification` on fallback-rendered images
    - _Requirements: 5.4_

  - [ ]* 5.4 Write unit tests for fallback renderer
    - Test fallback triggered when no valid images found
    - Test pdf2image invoked with correct DPI
    - Test preprocessing applied to fallback images
    - Test magnification applied to fallback images
    - Test error handling when fallback fails
    - Test logging of fallback invocation and failures
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.3, 8.5_

- [ ] 6. Integrate preprocessing and magnification into main extraction workflow
  - [x] 6.1 Update `extraer_de_pdf` to use preprocessing pipeline
    - Call `_apply_preprocessing` on all extracted images (both PyMuPDF and fallback)
    - Apply preprocessing before saving to disk
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 6.2 Update `extraer_de_pdf` to use magnification logic
    - Call `_apply_magnification` on all preprocessed images (both PyMuPDF and fallback)
    - Apply magnification after preprocessing, before saving to disk
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 6.3 Update `extraer_de_pdf` to use enhanced fallback renderer
    - Replace inline fallback code with call to `_fallback_render_page`
    - Ensure fallback images go through same preprocessing and magnification pipeline
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 6.4 Write property test for dimension filtering invariant
    - **Property 1: Dimension Filtering Invariant**
    - **Validates: Requirements 1.1, 1.2, 1.4**

  - [ ]* 6.5 Write property test for largest image selection
    - **Property 2: Largest Image Selection**
    - **Validates: Requirements 1.3**

- [x] 7. Checkpoint - Verify end-to-end extraction workflow
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement metadata and output validation
  - [ ]* 8.1 Write property test for PNG format invariant
    - **Property 8: PNG Format Invariant**
    - **Validates: Requirements 9.1**

  - [ ]* 8.2 Write property test for metadata structure invariant
    - **Property 9: Metadata Structure Invariant**
    - **Validates: Requirements 9.2, 10.5**

  - [ ]* 8.3 Write property test for filename convention invariant
    - **Property 10: Filename Convention Invariant**
    - **Validates: Requirements 9.3**

  - [ ]* 8.4 Write property test for output directory invariant
    - **Property 11: Output Directory Invariant**
    - **Validates: Requirements 9.4**

  - [ ]* 8.5 Write property test for metadata consistency across methods
    - **Property 12: Metadata Consistency Across Methods**
    - **Validates: Requirements 9.5**

- [ ] 9. Implement caption and page text preservation tests
  - [ ]* 9.1 Write unit tests for caption extraction preservation
    - Test spatial extraction using bbox (text below image)
    - Test fallback to first 500 chars when no text below
    - Test exclusion of text above image
    - Test preservation of existing `extraer_caption_imagen` static method
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 10.4_

  - [ ]* 9.2 Write property test for page text completeness
    - **Property 7: Page Text Completeness**
    - **Validates: Requirements 7.2**

  - [ ]* 9.3 Write unit tests for page text extraction
    - Test current page only (not adjacent pages)
    - Test error handling with empty string fallback
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 10. Implement error handling and logging tests
  - [ ]* 10.1 Write unit tests for error handling
    - Test PDF open failure (log error, return empty list)
    - Test PyMuPDF extraction failure (invoke fallback)
    - Test fallback renderer failure (log error, skip page, continue)
    - Test image validation failure (skip image, try next)
    - Test preprocessing failure (log warning, save unprocessed)
    - Test magnification failure (log warning, save original size)
    - Test caption extraction failure (use fallback)
    - Test OCR failure (store empty string)
    - Test page text extraction failure (store empty string)
    - Test file save failure (log error, skip image)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 10.2 Write unit tests for logging requirements
    - Test error logging format with page numbers
    - Test warning logging for preprocessing failures
    - Test info logging for fallback invocations
    - Test info logging for magnification operations
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 11. Verify backward compatibility
  - [ ]* 11.1 Write unit tests for backward compatibility
    - Test class name is ExtractorImagenesPDF
    - Test method signature for extraer_de_pdf(pdf_path: str) -> List[Dict[str, str]]
    - Test constructor signature __init__(directorio_salida: str)
    - Test static method extraer_caption_imagen signature preserved
    - Test metadata dictionary structure matches existing format
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ] 12. Integration testing with real PDFs
  - [ ]* 12.1 Write integration tests for end-to-end extraction
    - Test extraction from real PDF files (use pdf/arch2.pdf)
    - Test interaction with PyMuPDF library
    - Test interaction with pdf2image library
    - Test interaction with PIL/Pillow library
    - Test file system operations (save, directory creation)
    - Test complete workflow: extraction → validation → preprocessing → magnification → save → metadata
    - _Requirements: All requirements (end-to-end validation)_

- [ ] 13. Final checkpoint - Complete testing and validation
  - Run all property tests with minimum 100 iterations each
  - Run all unit tests and verify coverage
  - Run integration tests with real PDF samples
  - Verify all 12 correctness properties pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Documentation and code quality
  - [ ] 14.1 Add comprehensive docstrings to all new methods
    - Document `ImageExtractionConfig` class
    - Document `_apply_preprocessing` method
    - Document `_apply_magnification` method
    - Document `_fallback_render_page` method
    - Update `extraer_de_pdf` docstring with new workflow steps

  - [ ] 14.2 Add inline comments for complex logic
    - Comment magnification dimension calculation logic
    - Comment preprocessing enhancement ordering
    - Comment fallback invocation conditions

  - [ ] 14.3 Verify code style compliance
    - Follow PEP 8 style guide
    - Use type hints for all method signatures
    - Use descriptive variable names
    - Keep methods focused and single-purpose

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key integration points
- Property tests validate universal correctness properties (12 properties total)
- Unit tests validate specific examples, edge cases, and configuration behavior
- Integration tests validate end-to-end workflows with real PDFs
- All preprocessing and magnification logic applies to both PyMuPDF and fallback extraction paths
- Backward compatibility is critical - existing code must work without modification
- The implementation enhances the existing `ExtractorImagenesPDF` class in `ne4j-histo.py` (lines 1150-1300)

## Testing Strategy

**Property-Based Tests (hypothesis library):**
- Minimum 100 iterations per property test
- 12 correctness properties from design document
- Each property test includes comment referencing design property number

**Unit Tests:**
- Cover all branches and edge cases
- Test configuration behavior
- Test error handling and logging
- Test backward compatibility

**Integration Tests:**
- End-to-end extraction from real PDFs
- Verify interaction with external libraries
- Validate file system operations

## Implementation Order

1. **Phase 1 (Tasks 1-1.1):** Configuration infrastructure
2. **Phase 2 (Tasks 2-2.3):** Preprocessing pipeline
3. **Phase 3 (Tasks 3-3.5):** Magnification engine
4. **Phase 4 (Task 4):** Checkpoint - verify independent components
5. **Phase 5 (Tasks 5-5.4):** Enhanced fallback renderer
6. **Phase 6 (Tasks 6-6.5):** Integration into main workflow
7. **Phase 7 (Task 7):** Checkpoint - verify end-to-end workflow
8. **Phase 8 (Tasks 8-8.5):** Metadata and output validation
9. **Phase 9 (Tasks 9-9.3):** Caption and page text preservation
10. **Phase 10 (Tasks 10-10.2):** Error handling and logging
11. **Phase 11 (Tasks 11-11.1):** Backward compatibility verification
12. **Phase 12 (Tasks 12-12.1):** Integration testing
13. **Phase 13 (Task 13):** Final checkpoint and validation
14. **Phase 14 (Tasks 14-14.3):** Documentation and code quality
