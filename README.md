# A recommendation system paradigm based on Meetup database
This project is the final project at London Flatiron School. 
-- Project Status: On-Hold

## Project Intro/Objective
The purpose of this project is to recommend to Meetup members the personalised groups based on the members and groups descriptions. The impacts of this concept will be more control for members upon the recommendation system and an easier way for them to find relevant groups. This will lead to a higher attendance to events organised by the groups and a higher chance that people with the same interests will meet each other.

## Methods Used
Machine Learning

## Technologies
Python,
Pandas, Jupyter Notebook,
Gensim,
NLTK
 
## Project Description
For the project I used the data from Meetup API. The main challenge of this project is to sort and recommend the groups according to very specific interests people are. If I as a member I would like to find a group who is interested in impressionists art galleries, the system will give me this choices.
The workflow:
- The preprocessing stage.
- The processing stage: processing the text about members(bio) and about the groups(description) and to create a general Meetup context which will contain vectorised words. The tool for embed the words will be Word2Vec, with 200 neurons, 5 windows and 10 iterations.
- Generating the tags: extracting from the vocabulary the 20 most relevant words for each member bio and group description which could serve as tags. 
- Finding the distance between the members tags and the groups tags.
- Evaluating the algorithm through comparing the real data about the members attendance to groups and the recommended groups based on the algorithm described above. From these results it’s possible to compute the precision and recall which will give an idea about the algorithm accuracy.

## The problems appeared during the work:
- Missing data: from 1 million members just 400k had filled out their bio.
There are a lot of misspelled words which can’t be addressed even applying the lemmatization.
- During ‘distance between tags finding’ stage: because of a huge amount of computation 400 pairs per each member and group and low computer capacity, it wasn’t possible to interact all members and groups. The limited amount of computed data didn’t give me very trustfull evaluation results.
- Conceptually: the system is recommending the groups to members, but probably the groups contain more general information about the interests, but the description of the events they organise could contain more specific data, so this could be an improvement, to match the events tags with the members tags.
 
## Deliverables:
- Cleaning text Jupyter Notebook
- The algorithm Jupyter Notebook
- The algorithm functions (.py format)
- The presentation (pdf format)
 

