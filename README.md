# Overview

<p align="center">
  <img width="330" height="430" src="./System_overview.png">
</p>

The objective of this research is to generate a coherent and understandable text in Chinese.  
We extract commonsense knowledge from [ConceptNet](https://conceptnet.io/) automatically and select concepts by Monte-Carlo Tree Search (MCTS) algorithm.  
Combine text by human-crafted templates and use a constructed word embedding model and a Deep Neural Network (DNN) of discourse coherence model to evaluate and ensure the coherence of the generated text.

We evaluate generated text by human rating, and the [result](https://github.com/play0137/Generate_coherent_text/tree/master/evaluation) shows that it is more coherent when using the discourse coherence model.

## Chinese ConceptNet
Please refer to [Chinese ConceptNet](https://github.com/play0137/Chinese_ConceptNet).

## Models
- [Word embedding model](https://github.com/play0137/Traditional_Chinese_word_embedding#chinese-word-embeddings)
- [Discourse coherence model](https://mega.nz/file/vcAXzByB#qjZLRfwJ523rTlYvdY-h_gIFsk0hKzJNuiqHrBJDSx0) (download link)  
  A DNN model which is regarded as a reward function in MCTS.  
  The positive samples are the original paragraphs, and the negative samples are paragraphs with replacement of other connected concepts which has the same POS in ConceptNet.  
  Words in a sentence or a paragraph are closely related.
  Concepts replacement makes paragraph less coherent even a single replacement.  
  The training processes and experiments are in the [reports](https://github.com/play0137/Generate_coherent_text/tree/master/reports).

# Usage

- Download models above to the .\model folder
- Create a conda environment and install dependent packages with command  
  <code> conda create --name <env_name> --file requirements.txt </code>
- Run MCTS.py (It takes minutes to run the program.)
- Generated texts are in the .\output folder

# Reference
  
> Ying-Ren Chen (2021). [Generate coherent text using semantic embedding, common sense templates and Monte-Carlo tree search methods](https://etd.lib.nctu.edu.tw/cgi-bin/gs32/hugsweb.cgi?o=dnthucdr&s=id=%22G021040625840%22.&searchmode=basic) (Master's thesis, National Tsing Hua University, Hsinchu, Taiwan).  

BibTeX:  
> @mastersthesis{Chen:2021:generate_coherent_text,  
&nbsp;&nbsp;&nbsp;&nbsp; author = "Ying-Ren Chen,  
&nbsp;&nbsp;&nbsp;&nbsp; title = "Generate coherent text using semantic embedding, common sense templates and Monte-Carlo tree search methods",  
&nbsp;&nbsp;&nbsp;&nbsp; school = "National Tsing Hua University",  
&nbsp;&nbsp;&nbsp;&nbsp; pages = 136,  
&nbsp;&nbsp;&nbsp;&nbsp; year = 2021  
}  

  # License
  
  This work is licensed under a [GNU General Public License](https://www.gnu.org/licenses/gpl-3.0.html).
