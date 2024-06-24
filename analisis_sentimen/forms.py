from fileinput import FileInput
from django import forms

class UploadFileForm(forms.Form):
    data_csv = forms.FileField(
        widget= forms.FileInput(
            attrs = {
                'class':'form-control',
                'accept':'text/csv',
            }
        )
    )