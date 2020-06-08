from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired

from . import science

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a'


@app.route('/')
def index():
    form = GetRecipeForm(request.args)
    theurl = request.args.get(form.url.name)    
    if theurl:
        response = science.request_comparison(theurl)
        vec, features, recipes, fmproducts = science.load_data()
        new_features, percent = science.input_to_data(response, vec, fmproducts)
        similar = science.fetch_similar(new_features, features, vec, recipes, percent)
        return f'Your recipe is {percent} % local, Similar recipes: {similar}'   

    return render_template("index.html", form=form)





class GetRecipeForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])
    submit = SubmitField('Get Local!')
