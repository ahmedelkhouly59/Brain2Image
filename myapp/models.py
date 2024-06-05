from django.db import models

# Create your models here.
class Deployment(models.Model):
   Your_Name = models.CharField(max_length=200)
   Email_Address=models.EmailField(max_length=200)    
   File_To_Upload= models.FileField(default='hi')

   def __str__(self):
      return self.name