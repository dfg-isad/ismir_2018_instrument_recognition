# Complementary website

## Publication

``` bibtex
@InProceedings{Gomez:2018:ISMIR,
	author = {Juan Gomez and Jakob Abe{\ss}er and Estefan{'i}a Cano},
	title = {Jazz Solo Instrument Classification with Convolutional Neural Networks, Source Separation, and Transfer Learning},
	year = {2018},
  booktitle = {Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR)},
	address = {Paris, France},
}
```
## Content

This page provides metadata to reconstruct the MONOTIMBRAL and JAZZ dataset from the abovementioned publication.

### MONOTIMBRAL dataset

The file monotimbral_dataset.csv contains a list of youtube URLs, segment information for 16 different musical instruments (monotimbral recordings, i.e., monophonic of polyphonic instrument recordings without overlap to other instruments). There are 30 recordings per instruments.

### JAZZ dataset

The JAZZ dataset was compiled from solos from the Weimar Jazz Database (WJD) (https://jazzomat.hfm-weimar.de/dbformat/dboverview.html) and additional jazz solo excerpts taken from youtube videos.
The file jazz_dataset.csv lists all the files in the dataset with their source (either WJD or YOUTUBE), the file jazz_dataset_youtube.csv lists the youtube URLs and metadata for the additional files taken from Youtube.

## Comment 

* Youtube URLs can become obsolete after a while if the original video is removed

## Contact

* Please contact jakob.abesser[at]idmt.fraunhofer.de for further questions
