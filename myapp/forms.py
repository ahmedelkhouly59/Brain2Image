from .models import Deployment
from django.forms import ModelForm

class DeploymentForm(ModelForm):
    class Meta:
        model=Deployment
        fields='__all__'