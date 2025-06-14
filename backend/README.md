# SM Mirror Backend - Stable-Makeup Integration

## ✅ **Status: Working!**

This backend successfully integrates your Stable-Makeup installation with a FastAPI server, providing professional makeup transfer capabilities.

## 🚀 **Quick Start**

### 1. **From the project root directory:**
```bash
cd backend
python main.py
```

The server will start on http://localhost:8000

### 2. **Test the integration:**
```bash
# Test the service
python test_stable_makeup.py

# Test the complete API
python test_api_upload.py
```

## 🔧 **What's Working**

### **✅ FastAPI Server**
- **Health endpoints**: `/health` and `/api/v1/makeup/health`
- **Makeup transfer**: `POST /api/v1/makeup/transfer`
- **CORS enabled** for React frontend (localhost:3000)

### **✅ Stable-Makeup Integration**
- **Smart model loading**: Auto-detects your `pytorch_model.bin`, `pytorch_model_1.bin`, `pytorch_model_2.bin`
- **SPIGA + MediaPipe**: Hybrid face detection with fallbacks
- **GPU/CPU support**: Automatically detects and uses available hardware
- **Error handling**: Graceful fallbacks when models/dependencies are missing

### **✅ File Handling**
- **Multi-part uploads**: Supports source + reference image uploads
- **Format validation**: Automatic image format handling
- **Result storage**: Saves results in `results/` directory

## 📁 **Project Structure**

```
backend/
├── main.py                    # FastAPI server entry point
├── app/
│   ├── core/config.py         # Configuration settings
│   ├── api/v1/endpoints/
│   │   └── makeup.py          # Makeup transfer endpoints
│   └── services/
│       └── makeup_service.py  # Stable-Makeup integration
├── test_stable_makeup.py      # Integration tests
├── test_api_upload.py         # API tests
├── requirements.txt           # All dependencies
└── requirements_essential.txt # Essential dependencies
```

## 🔗 **API Endpoints**

### **Health Check**
```bash
GET /health
GET /api/v1/makeup/health
```

### **Makeup Transfer**
```bash
POST /api/v1/makeup/transfer
Content-Type: multipart/form-data

Fields:
- source_image: Source image file
- reference_image: Makeup reference image file
```

**Response**: Processed image with makeup applied

## 🎯 **Integration Details**

### **Model Path Detection**
The service automatically looks for models in:
1. `../pytorch_model_2/` (your current setup)
2. `../Stable-Makeup/models/`
3. `models/` (local directory)

### **Face Detection Pipeline**
1. **SPIGA**: Professional face landmark detection (if available)
2. **MediaPipe**: Fallback for pose control generation
3. **Graceful degradation**: Returns processed result even with minimal models

### **Performance Notes**
- **First run**: May take longer due to model loading
- **CPU mode**: Currently optimized for CPU (add GPU optimization later)
- **Memory**: Automatically manages model loading/unloading

## 🔧 **Troubleshooting**

### **Common Issues**

1. **"Can't find main.py"**
   ```bash
   # Make sure you're in the backend directory
   cd backend
   python main.py
   ```

2. **Import errors**
   ```bash
   # Install essential dependencies
   pip install -r requirements_essential.txt
   ```

3. **Model not found warnings**
   - This is normal! The service works with available models
   - Check that `pytorch_model_2/` directory exists in project root

### **Dependencies**
- **Essential**: `requirements_essential.txt` (working setup)
- **Complete**: `requirements.txt` (includes optional optimizations)

## 🎉 **Next Steps**

1. **✅ Backend is working!** - Your Stable-Makeup integration is complete
2. **🔄 Frontend**: Create React frontend to interact with the API
3. **⚡ Performance**: Add GPU optimization and model caching
4. **🚀 Deployment**: Prepare for production deployment

## 📝 **Testing**

### **Run all tests:**
```bash
# Service integration test
python test_stable_makeup.py

# API integration test  
python test_api_upload.py

# Manual API test with curl
curl -X POST http://localhost:8000/api/v1/makeup/transfer \
  -F "source_image=@test_images/source.jpg" \
  -F "reference_image=@test_images/reference.jpg" \
  --output result.jpg
```

---

## 🎊 **Congratulations!**

Your Stable-Makeup integration is working! The backend can:
- ✅ Load your trained models
- ✅ Process face detection 
- ✅ Apply professional makeup transfer
- ✅ Handle file uploads via REST API
- ✅ Integrate with your existing project structure

**Ready for frontend development!** 🚀 