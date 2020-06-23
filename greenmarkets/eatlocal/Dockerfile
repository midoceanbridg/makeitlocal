FROM conda/miniconda3 


RUN conda config --append channels conda-forge
RUN conda install -c intel scikit-learn
RUN conda install numpy fuzzywuzzy gensim 
RUN pip install requests flask flask_wtf
COPY generated_data /generated_data 
RUN conda install pandas
RUN conda install nltk
RUN mkdir -p /root/nltk_data/corpora/
RUN mkdir -p  /root/nltk_data/taggers/averaged_perceptron_tagger/
RUN mv generated_data/wordnet/ /root/nltk_data/corpora/
RUN mv generated_data/averaged_perceptron_tagger.pickle /root/nltk_data/taggers/averaged_perceptron_tagger/ 
ENV LH_GENDIR=/generated_data
COPY thelocalhost /thelocalhost
ENV FLASK_APP=/thelocalhost/app.py
CMD ["flask", "run", "-h", "0.0.0.0"]

