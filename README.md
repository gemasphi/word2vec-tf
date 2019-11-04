# Word2Vec-tf
A word2vec implementation in tensorflow using the CBOW architecture that uses a generator to read the dataset. Included in the class is also a way to evaluate the word embedding quality on WordSim353 using the spearmans correlation coefficient.

A retrofit[1] implementation is also included in the class to further improve the embeddings. 


    [1] @misc{faruqui2014retrofitting,
        title={Retrofitting Word Vectors to Semantic Lexicons},
        author={Manaal Faruqui and Jesse Dodge and Sujay K. Jauhar and Chris Dyer and Eduard Hovy and Noah A. Smith},
        year={2014},
        eprint={1411.4166},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
      } 
