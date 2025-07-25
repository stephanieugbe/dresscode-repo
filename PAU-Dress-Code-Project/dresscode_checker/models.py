from django.db import models
from django.utils import timezone
import uuid

class ComplianceCheck(models.Model):
    COMPLIANCE_CHOICES = [
        ('compliant', 'Compliant'),
        ('non_compliant', 'Non-Compliant'),
        ('error', 'Error'),
    ]
    
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    compliance_result = models.CharField(max_length=20, choices=COMPLIANCE_CHOICES)
    confidence_score = models.FloatField(null=True, blank=True)
    violations = models.JSONField(default=list, blank=True)
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, default='female')
    timestamp = models.DateTimeField(default=timezone.now)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Compliance Check'
        verbose_name_plural = 'Compliance Checks'
    
    def __str__(self):
        return f"Check {str(self.id)[:8]} - {self.compliance_result} ({self.gender})"