from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
from typing import Optional
from app.core.config import settings
from app.services.makeup_service import MakeupService

router = APIRouter()
makeup_service = MakeupService()

@router.post("/transfer")
async def makeup_transfer(
    source_image: UploadFile = File(..., description="Source face image"),
    reference_image: UploadFile = File(..., description="Makeup reference image")
):
    """
    Apply makeup transfer from reference image to source image
    """
    try:
        # Validate file types
        if not source_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Source file must be an image")
        if not reference_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Reference file must be an image")
        
        # Generate unique filenames
        source_filename = f"{uuid.uuid4()}_source.jpg"
        reference_filename = f"{uuid.uuid4()}_reference.jpg"
        result_filename = f"{uuid.uuid4()}_result.jpg"
        
        # Save uploaded files
        source_path = os.path.join(settings.UPLOAD_DIR, source_filename)
        reference_path = os.path.join(settings.UPLOAD_DIR, reference_filename)
        result_path = os.path.join(settings.RESULTS_DIR, result_filename)
        
        with open(source_path, "wb") as buffer:
            content = await source_image.read()
            buffer.write(content)
        
        with open(reference_path, "wb") as buffer:
            content = await reference_image.read()
            buffer.write(content)
        
        # Process makeup transfer
        success = await makeup_service.apply_makeup_transfer(
            source_path, reference_path, result_path
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Makeup transfer failed")
        
        # Return result image
        return FileResponse(
            result_path,
            media_type='image/jpeg',
            filename=f"makeup_result_{uuid.uuid4().hex[:8]}.jpg"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@router.get("/health")
def health_check():
    """Health check for makeup service"""
    return {"status": "healthy", "service": "makeup-transfer"} 