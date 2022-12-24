from django import forms

# creating a form
class InputForm(forms.Form):

    phrase = forms.CharField(widget=forms.Textarea(attrs={'class': 'form-control'}))
    CHOICES = [('kk','Kazakh'), ('ba','Bashkir'), ('tt','Tatar')]
    language = forms.CharField(label='Language', widget=forms.RadioSelect(choices=CHOICES))
    