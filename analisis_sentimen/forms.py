from fileinput import FileInput
from pydoc import classname
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

class ParameterSVMForm(forms.Form):
    kernel = forms.CharField(
        widget= forms.TextInput(
            attrs={
                'class':'form-select'
            }
        )
    )
    nilai_C = forms.FloatField(
        widget= forms.NumberInput(
             attrs= {
                'class':'form-control',
                'placeholder':'Nilai C'
            }
        )
    ) 
    gamma = forms.FloatField(
        widget= forms.NumberInput(
            attrs= {
                'class':'form-control',
                'placeholder':'Gamma'
            }
        )
    )
    degree = forms.IntegerField(
        widget= forms.NumberInput(
            attrs= {
                'class':'form-control',
                'placeholder':'Degree'
            }
        )
    )