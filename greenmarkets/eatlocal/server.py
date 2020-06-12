from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

from . import localeats_twostage

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a'


@app.route('/')
def index():
    form = GetRecipeForm(request.args)
    theurl = request.args.get(form.url.name) 
    allout = None   
    if theurl:
        ingredients = localeats_twostage.request_comparison(theurl)
        w2vm, aisledict, noise, atFM, FMinfo = localeats_twostage.load_data()
        noise_free_ing = localeats_twostage.removenoise(ingredients, noise)
        allout = localeats_twostage.rulesofsimilarity(noise_free_ing, w2vm, aisledict, atFM, FMinfo)
        

        
    return render_template("index.html", form=form, allout=allout)





class GetRecipeForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])
    submit = SubmitField('Get Local!')
