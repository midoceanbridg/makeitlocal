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

        if cur_rec is not None:  
            allout, wheretoshop = localeats_twostage.get_results(ingredients, cur_rec)      
            return render_template("indexmountain.html.j2", form=form, allout=allout, wheretoshop=wheretoshop)
        else:
            return render_template("404.html.j2", form=form) 
    else:
        return render_template("indexmountain.html.j2", form=form, allout=allout, wheretoshop=wheretoshop) 



class GetRecipeForm(FlaskForm):
    url = StringField('URL', validators=[DataRequired()])
    submit = SubmitField('Get Local!')

if __name__ == "__main__":
    app.run()
