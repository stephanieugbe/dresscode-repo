from django.contrib import admin
from .models import ComplianceCheck

@admin.register(ComplianceCheck)
class ComplianceCheckAdmin(admin.ModelAdmin):
    list_display = ['id', 'compliance_result', 'confidence_score', 'timestamp', 'ip_address']
    list_filter = ['compliance_result', 'timestamp']
    search_fields = ['id', 'ip_address']
    readonly_fields = ['id', 'timestamp']
    ordering = ['-timestamp']
    
    def has_add_permission(self, request):
        return False