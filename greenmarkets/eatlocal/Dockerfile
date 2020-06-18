FROM conda/miniconda3 


RUN conda config --append channels conda-forge
RUN conda install -c intel scikit-learn
RUN conda install numpy fuzzywuzzy gensim 
RUN pip install requests flask flask_wtf
COPY generated_data /generated_data 
RUN conda install pandas
ENV LH_GENDIR=/generated_data
COPY thelocalhost /thelocalhost
ENV FLASK_APP=/thelocalhost/app.py
CMD ["flask", "run", "-h", "0.0.0.0"]
