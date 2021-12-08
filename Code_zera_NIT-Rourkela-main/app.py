import numpy as np
import pandas as pd
import re
from flask import Flask, render_template, request, redirect
from pandas.core.frame import DataFrame
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer



data= pd.read_csv("cosmetics.csv")
data

categories=data["Label"].values
categories_unique=data["Label"].unique()

#writingg the brands down
brands=data["Brand"].values
brands_unique=data["Brand"].unique()

#columns of the parameters like combination, moist, dry etc
para_columns= data.columns[6:]

#column storing the composition
composition=data.Ingredients.values

import re
text=data.Ingredients.values
def text_cleaner(text):
  #ct = re.sub('[^a-zA-Z]', ' ', text)
  ct=text.lower()
  ct=ct.split(",")
  ct=sorted(set(ct))
  return ct

def cleaner():
  text=data.Ingredients.values
  clean_text=[]
  for i in text:
    clean_text.append(text_cleaner(i))
  text=clean_text
  return text
text=cleaner()

data=data.drop("Ingredients", axis=1)
data["Ingredients"]=text
data["Composition"]=composition
data.to_csv('processed_data.csv', header=True, index=False)

data = pd.read_csv('processed_data.csv')

def filter(category="Moisturizer", price_low=0, price_high=max(data.Price),comb="No", dry="No", normal="No", oily="No", sensitive="No"):
  df1 =data[(data["Label"] == category)  &  (data["Price"]>=price_low ) & (data["Price"]<=price_high)].reset_index().drop("index", axis=1) 
  comb_var=0
  dry_var=0
  norm_var=0
  oil_var=0
  sen_var=0
  if comb=="Yes":
    comb_var=1
  if dry=="Yes":
    dry_var=1
  if normal=="Yes":
    norm_var=1
  if oily=="Yes":
    oil_var=1
  if sensitive=="Yes":
    sen_var=1
  df2 = df1[(df1["Combination"]==comb_var)^(df1["Dry"]==dry_var)^(df1["Normal"]==norm_var)^(df1["Oily"]==oil_var)^(df1["Sensitive"]==sen_var)].reset_index().drop("index", axis=1).iloc[:,0:4]
  return df2


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Todo(db.Model):
    sno = db.Column(db.Integer, primary_key = True)
    title = db.Column(db.String(200), nullable = True)
    desc = db.Column(db.String(200), nullable = True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"




@app.route("/", methods=['GET', 'POST'])
def hello_world():
    tdf=[]
    if request.method=='POST':
        #print(request.form['title'])
        title = request.form['title']
        minp = int(request.form['minp'])
        maxp = int(request.form['maxp'])
        comb = request.form['comb']
        dry = request.form['dry']
        normal = request.form['normal']
        oily = request.form['oily']
        sensitive = request.form['sensitive']
        #desc = request.form['desc']
        df = filter(title,minp,maxp,comb,dry,normal,oily,sensitive)
        print(df)
        tdf= df.values.tolist()
    #     todo = Todo(title=title, desc=desc)
    #     db.session.add(todo)
    #     db.session.commit()
    # allTodo = Todo.query.all()

    #return render_template('index.html', allTodo=allTodo, tdf=tdf)
    return render_template('index.html', tdf=tdf)
    #return "<p>Hello, World!</p>"



# @app.route("/show")
# def products():
#     allTodo = Todo.query.all()
#     print(allTodo)
#     return "<p>This is product page</p>"

data1 = pd.read_csv("cosmetics.csv")

cleaned_data = data1.drop(columns=['Combination', 'Dry', 'Oily', 'Sensitive','Normal'])


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
cleaned_data['Ingredients'] = cleaned_data['Ingredients'].fillna('')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(cleaned_data['Ingredients'])



# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and titles
indices = pd.Series(cleaned_data.index, index=cleaned_data['Name']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    product_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return cleaned_data.iloc[product_indices]

@app.route("/show", methods=['GET', 'POST'])
def products():
    tdf2=[]
    if request.method=='POST':
        desc = request.form['desc']
        df2 = give_rec(desc)
        tdf2= df2.values.tolist()
    return render_template('product.html', tdf2=tdf2)


if __name__ == "__main__":
    app.run(debug=True, port=8000)