from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

from . import localeats_twostage

app = Flask(__name__, static_url_path='')
app.config['SECRET_KEY'] = 'a'


@app.route('/')
def index():
    form = GetRecipeForm(request.args)
    theurl = request.args.get(form.url.name) 
    allout = None   
    wheretoshop = None
    if theurl:
       ingredients, cur_rec = localeats_twostage.request_comparison(theurl)
       w2vm, aisledict, noise, atFM, FMinfo, ingvect, ingfeatures, fulling, recvect, recfeatures, recdoc = localeats_twostage.load_data()
       noise_free_ing = localeats_twostage.removenoise(ingredients, noise)
       allout, wheretoshop = localeats_twostage.rulesofsimilarity(noise_free_ing, w2vm, aisledict, atFM, FMinfo)
       localeats_twostage.validationstep(allout, fulling, ingvect, ingfeatures, recvect, recfeatures, recdoc, cur_rec) 

        
    return render_template("indexmountain.html.j2", form=form, allout=allout, wheretoshop=wheretoshop)





class GetRecipeForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])
    submit = SubmitField('Get Local!')
