# TextSummarization-MMR

Fork modifying this code to work for French

MMR (Maximum Marginal Relevance) is an extractive summarization that was introduced by Jaime Carbonell and Jade Goldstein.
http://repository.cmu.edu/cgi/viewcontent.cgi?article=1330&context=compsci

MMR aims to obtain the most relevance sentences by scoring whole sentences in the document.
The MMR criterion strives to reduce redudancy while maintaining the content relevance in re-ranking retireved sentences

## Dependency

- This code is implemented in Python
- Stemmer from nltk library ```pip install nltk``` (dont forget to install nltk's resources)
- The list of stopwords comes from https://github.com/stopwords-iso/stopwords-iso/ 
- It also use "sklearn library". Please install it by ```pip install sklearn```
- And this one ```pip install termcolor```

## How to run

- In you terminal type it as ```python mmr.py [Document.txt]```

## Output

![MMR Output](https://github.com/fajri91/document/blob/master/mmr.jpg)
