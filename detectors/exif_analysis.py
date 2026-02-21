import PIL.Image
from PIL.ExifTags import TAGS

def extract_exif(image_path):
    exif_data = {}
    try:
        image = PIL.Image.open(image_path)
        exif = image.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = str(value)
    except Exception:
        pass
    return exif_data

def check_camera_metadata(exif):
    camera_fields = ['Make', 'Model', 'ExposureProgram', 'ISOSpeedRatings', 'DateTimeOriginal']
    found = sum(1 for field in camera_fields if field in exif)
    return found

def detect_tampering(exif):
    software = exif.get('Software', '').lower()
    suspicious = ['photoshop', 'gimp', 'stable diffusion', 'midjourney', 'dall-e']
    if any(s in software for s in suspicious) and check_camera_metadata(exif) == 0:
        return True
    return False

def compute_metadata_score(image_path):
    """
    Returns float [0-1].
    Real images: Complete EXIF -> 0.7-1.0
    Edited real: Partial EXIF -> 0.4-0.6
    AI / Stripped: No EXIF -> 0.0-0.3
    """
    exif = extract_exif(image_path)
    if not exif:
        return 0.1  # No EXIF
    
    if detect_tampering(exif):
        return 0.2  # AI/Edited without camera data
        
    camera_fields_found = check_camera_metadata(exif)
    if camera_fields_found >= 3:
        return 0.9  # Strong confidence it's from a camera
    elif camera_fields_found > 0:
        return 0.6  # Partial camera data
    
    return 0.3  # Has EXIF but no camera signatures
