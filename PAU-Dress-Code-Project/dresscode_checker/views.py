from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os
import tempfile
import json

# PyTorch imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np

from .models import ComplianceCheck

def get_client_ip(request):
    
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def load_pytorch_model(gender='female'):
    
    try:
        model_filename = f'pau_dresscode_resnet_{gender}.pth'
        model_path = os.path.join(settings.BASE_DIR, 'models', model_filename)
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Using dummy predictions.")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if gender == 'male':
            class DressCodeClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = models.resnet18(pretrained=False)
                    self.backbone.fc = nn.Sequential(
                        nn.Dropout(0.5),           
                        nn.Linear(512, 128),       
                        nn.ReLU(),                 
                        nn.Dropout(0.3),           
                        nn.Linear(128, 2)          
                    )
                
                def forward(self, x):
                    return self.backbone(x)
            
            model = DressCodeClassifier()
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            
        else:  
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, 2)
            )
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        print(f"✅ {gender.capitalize()} model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading {gender} model: {e}. Using dummy predictions.")
        return None
        

def preprocess_image_pytorch(image_path):
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  
        
        return image_tensor
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def analyze_violations(image_path, compliance_result, gender='female'):
    
    violations = []
    
    return violations

def predict_compliance(image_path, gender='female'):
    
    model = load_pytorch_model(gender)
    
    if model is None:
        import random
        compliance = random.choice(['compliant', 'non_compliant'])
        confidence = random.uniform(0.75, 0.95)
        
        violations = analyze_violations(image_path, compliance, gender)
        
        return {
            'compliance': compliance,
            'confidence': confidence,
            'violations': violations,
            'model_used': f'dummy_{gender}',
            'gender': gender
        }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        with torch.no_grad():
            processed_image = preprocess_image_pytorch(image_path)
            processed_image = processed_image.to(device)
            
            outputs = model(processed_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            compliance = 'compliant' if predicted.item() == 0 else 'non_compliant'
            confidence_score = confidence.item()
            
            violations = analyze_violations(image_path, compliance, gender)
            
            return {
                'compliance': compliance,
                'confidence': confidence_score,
                'violations': violations,
                'model_used': f'pytorch_resnet_{gender}',
                'gender': gender
            }
            
    except Exception as e:
        return {
            'compliance': 'error',
            'confidence': 0.0,
            'violations': [f'Processing error: {str(e)}'],
            'model_used': f'error_{gender}',
            'gender': gender
        }

# Main Views
def home(request):
    
    total_checks = ComplianceCheck.objects.count()
    recent_checks = ComplianceCheck.objects.order_by('-timestamp')[:5]
    
    context = {
        'total_checks': total_checks,
        'recent_checks': recent_checks,
    }
    return render(request, 'dresscode_checker/home.html', context)

def check_outfit(request):
    
    return render(request, 'dresscode_checker/check.html')

def guidelines(request):
    
    return render(request, 'dresscode_checker/guidelines.html')

def about(request):
    
    return render(request, 'dresscode_checker/about.html')

@csrf_exempt
@require_http_methods(["POST"])
def api_check_compliance(request):
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image uploaded'}, status=400)
        
        if 'gender' not in request.POST:
            return JsonResponse({'error': 'No gender specified'}, status=400)
        
        uploaded_file = request.FILES['image']
        gender = request.POST.get('gender')
        
        if gender not in ['male', 'female']:
            return JsonResponse({'error': 'Invalid gender specified. Must be male or female.'}, status=400)
        
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({
                'error': 'Invalid file type. Please upload JPG or PNG images only.'
            }, status=400)
        
        if uploaded_file.size > 16 * 1024 * 1024:
            return JsonResponse({
                'error': 'File too large. Please upload images smaller than 16MB.'
            }, status=400)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        
        try:
            result = predict_compliance(temp_file_path, gender)
            
            compliance_check = ComplianceCheck.objects.create(
                compliance_result=result['compliance'],
                confidence_score=result['confidence'],
                violations=result['violations'],
                gender=gender,
                ip_address=get_client_ip(request)
            )
            
            return JsonResponse({
                'success': True,
                'result': {
                    'compliance': result['compliance'],
                    'confidence': round(result['confidence'] * 100, 2),
                    'violations': result['violations'],
                    'check_id': str(compliance_check.id),
                    'timestamp': compliance_check.timestamp.isoformat(),
                    'model_used': result.get('model_used', 'unknown'),
                    'gender': result.get('gender', gender)
                }
            })
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        return JsonResponse({
            'error': f'Processing error: {str(e)}'
        }, status=500)

def api_stats(request):
    try:
        total_checks = ComplianceCheck.objects.count()
        compliant_checks = ComplianceCheck.objects.filter(compliance_result='compliant').count()
        non_compliant_checks = ComplianceCheck.objects.filter(compliance_result='non_compliant').count()
        error_checks = ComplianceCheck.objects.filter(compliance_result='error').count()
        
        female_checks = ComplianceCheck.objects.filter(gender='female').count()
        male_checks = ComplianceCheck.objects.filter(gender='male').count()
        
        compliance_rate = (compliant_checks / total_checks * 100) if total_checks > 0 else 0
        
        from django.utils import timezone
        from datetime import timedelta
        
        yesterday = timezone.now() - timedelta(days=1)
        recent_checks = ComplianceCheck.objects.filter(timestamp__gte=yesterday).count()
        
        return JsonResponse({
            'total_checks': total_checks,
            'compliant_checks': compliant_checks,
            'non_compliant_checks': non_compliant_checks,
            'error_checks': error_checks,
            'female_checks': female_checks,
            'male_checks': male_checks,
            'compliance_rate': round(compliance_rate, 2),
            'recent_checks_24h': recent_checks,
            'timestamp': timezone.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error retrieving statistics: {str(e)}'
        }, status=500)

def api_recent_checks(request):
    try:
        recent_checks = ComplianceCheck.objects.order_by('-timestamp')[:10]
        
        checks_data = []
        for check in recent_checks:
            checks_data.append({
                'id': str(check.id),
                'gender': check.gender,
                'compliance_result': check.compliance_result,
                'confidence_score': check.confidence_score,
                'violations_count': len(check.violations) if check.violations else 0,
                'timestamp': check.timestamp.isoformat(),
                'ip_address': check.ip_address[:10] + '...' if check.ip_address else None  # Partial IP for privacy
            })
        
        return JsonResponse({
            'recent_checks': checks_data,
            'count': len(checks_data)
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error retrieving recent checks: {str(e)}'
        }, status=500)
    
